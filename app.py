"""
NITRO NEBULA  ·  The Cosmic MPG Oracle
--------------------------------------
End-to-end Automobile Fuel-Efficiency prediction app for Streamlit.

Pipeline
  1. Data extraction      -> load auto-mpg CSV (repo file or user upload)
  2. Cleaning             -> normalise headers, '?' horsepower -> NaN, drop dupes / bad targets
  3. Feature engineering  -> car age, power-to-weight, displacement per cylinder
  4. Missing values       -> median imputation (numeric), most-frequent (categorical)
  5. Encoding             -> One-Hot encoding of `origin` (USA / Europe / Japan)
  6. Scaling              -> StandardScaler on all numeric features
  7. Modelling            -> Ridge, SVR, Random Forest, Gradient Boosting compared by 5-fold CV
  8. Serving              -> best model is refit on all data and used for live predictions

Run locally:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIG
# ──────────────────────────────────────────────────────────────────────────────
APP_NAME = "NITRO NEBULA"
APP_TAGLINE = "The Cosmic MPG Oracle"
RANDOM_STATE = 42
REF_YEAR = 83  # reference year for the "car age" feature (data spans 1970-1982)

RAW_FEATURES = ["cylinders", "displacement", "horsepower", "weight",
                "acceleration", "model year", "origin"]
NUMERIC_FEATURES = ["cylinders", "displacement", "horsepower", "weight",
                    "acceleration", "car_age", "power_to_weight", "disp_per_cyl"]
CATEGORICAL_FEATURES = ["origin"]
ORIGIN_MAP = {1: "🇺🇸 USA", 2: "🇪🇺 Europe", 3: "🇯🇵 Japan"}

MPG_TO_KML = 0.425144       # US mpg -> km per litre
MPG_TO_L100 = 235.215       # US mpg -> litres per 100 km


# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
def find_dataset():
    """Look for an auto-mpg CSV next to app.py."""
    here = Path(__file__).resolve().parent
    for p in sorted(here.glob("*.csv")):
        if "mpg" in p.name.lower():
            return p
    return None


def load_raw(source):
    """Read the CSV from a path or an uploaded file object."""
    df = pd.read_csv(source)
    df.columns = [c.strip().lower().replace("_", " ") for c in df.columns]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. CLEANING (row-level, target-aware)  &  3. FEATURE ENGINEERING (in-pipeline)
# ──────────────────────────────────────────────────────────────────────────────
def clean_dataset(df):
    """Fix dtypes, convert '?' to NaN, remove duplicates and rows without a target."""
    df = df.copy()
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")  # '?' -> NaN
    for col in ["mpg", "cylinders", "displacement", "weight", "acceleration",
                "model year", "origin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates().dropna(subset=["mpg"]).reset_index(drop=True)
    return df


def engineer_features(X):
    """Runs INSIDE the sklearn pipeline, so raw inputs from the UI work directly."""
    X = X.copy()
    X["horsepower"] = pd.to_numeric(X["horsepower"], errors="coerce")
    X["car_age"] = REF_YEAR - X["model year"]
    X["power_to_weight"] = X["horsepower"] / X["weight"] * 1000
    X["disp_per_cyl"] = X["displacement"] / X["cylinders"]
    return X


def build_pipeline(model):
    """Impute -> scale numerics, impute -> one-hot categoricals, then the regressor."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        sparse_threshold=0,
    )
    return Pipeline([
        ("features", FunctionTransformer(engineer_features)),
        ("preprocess", preprocess),
        ("model", model),
    ])


def candidate_models():
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Support Vector (RBF)": SVR(C=10, epsilon=0.5),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            random_state=RANDOM_STATE),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. TRAINING  (cached – runs once per server session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_everything(raw_df):
    df = clean_dataset(raw_df)
    X, y = df[RAW_FEATURES], df["mpg"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, model in candidate_models().items():
        pipe = build_pipeline(model)
        scores = cross_validate(
            pipe, X_tr, y_tr, cv=cv,
            scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"})
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "Model": name,
            "CV R²": scores["test_r2"].mean(),
            "CV RMSE": -scores["test_rmse"].mean(),
            "Test R²": r2_score(y_te, pred),
            "Test RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
            "Test MAE": mean_absolute_error(y_te, pred),
        })
        fitted[name] = pipe

    board = pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True)
    best_name = board.loc[0, "Model"]
    best_on_train = fitted[best_name]
    holdout_pred = best_on_train.predict(X_te)

    imp = permutation_importance(best_on_train, X_te, y_te, n_repeats=15,
                                 random_state=RANDOM_STATE, scoring="r2")
    importance = (pd.DataFrame({"Feature": RAW_FEATURES, "Importance": imp.importances_mean})
                  .sort_values("Importance"))

    # Serving model: best architecture refit on 100 % of the data
    final = clone(build_pipeline(candidate_models()[best_name])).fit(X, y)

    # "Cars like yours" lookup lives in the scaled feature space
    Xt = final[:-1].transform(X)
    knn = NearestNeighbors(n_neighbors=3).fit(Xt)

    return {
        "df": df, "final": final, "best_name": best_name, "board": board,
        "y_te": y_te.to_numpy(), "holdout_pred": holdout_pred,
        "importance": importance, "knn": knn,
        "n_missing_hp": int(df["horsepower"].isna().sum()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. UI HELPERS – theme, animations, gauge
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
  background: linear-gradient(135deg, #0b0221, #1a0b3d, #0a2a4a, #2b0a3d);
  background-size: 400% 400%;
  animation: nebula 22s ease infinite;
  color: #eaf6ff;
}
@keyframes nebula { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

/* twinkling star-field */
.stApp::before {
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.55;
  background-image:
    radial-gradient(1.5px 1.5px at 12% 18%, #fff, transparent),
    radial-gradient(1px 1px at 28% 72%, #9ff, transparent),
    radial-gradient(2px 2px at 44% 34%, #fff, transparent),
    radial-gradient(1px 1px at 63% 81%, #f9f, transparent),
    radial-gradient(1.5px 1.5px at 78% 22%, #fff, transparent),
    radial-gradient(1px 1px at 91% 58%, #9ff, transparent),
    radial-gradient(1px 1px at 6% 90%, #fff, transparent),
    radial-gradient(2px 2px at 52% 8%, #ffd27f, transparent);
  animation: twinkle 4s ease-in-out infinite alternate;
}
@keyframes twinkle { from{opacity:.25} to{opacity:.75} }

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(20,8,50,.92), rgba(8,25,55,.92));
  border-right: 1px solid rgba(0,229,255,.25);
}

/* hero */
.hero { text-align:center; padding: 8px 0 0 0; animation: fadeUp .9s ease both; }
.hero h1 {
  font-family:'Orbitron', sans-serif; font-weight:900; letter-spacing:.14em;
  font-size: clamp(2rem, 5vw, 3.6rem); margin:0;
  background: linear-gradient(90deg,#00e5ff,#ff2bd6,#ffb300,#00e5ff);
  background-size:300% 100%; -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; animation: shine 6s linear infinite;
  filter: drop-shadow(0 0 14px rgba(255,43,214,.45));
}
@keyframes shine { to { background-position: 300% 0; } }
.hero p { letter-spacing:.35em; text-transform:uppercase; color:#9fd8ff; font-size:.8rem; margin:.2rem 0 0 0; }

/* animated road */
.road { position:relative; height:46px; margin:10px auto 18px auto; max-width:900px; overflow:hidden;
  border-radius:10px; background:linear-gradient(#17082e,#0b1230); border:1px solid rgba(0,229,255,.25); }
.road::before { content:""; position:absolute; left:0; right:0; top:50%; height:3px;
  background:repeating-linear-gradient(90deg,#ffb300 0 28px,transparent 28px 56px);
  animation: dash 1.1s linear infinite; }
@keyframes dash { from{background-position:0 0} to{background-position:-56px 0} }
.car { position:absolute; top:2px; font-size:30px; animation: drive 7s linear infinite; }
@keyframes drive { from{left:-60px} to{left:105%} }

@keyframes fadeUp { from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:none} }

/* glass cards */
.card {
  background: rgba(255,255,255,.05); border:1px solid rgba(0,229,255,.28);
  border-radius:16px; padding:16px 18px; backdrop-filter: blur(8px);
  box-shadow: 0 0 22px rgba(0,229,255,.08); animation: fadeUp .8s ease both;
  transition: transform .25s, box-shadow .25s;
}
.card:hover { transform: translateY(-4px); box-shadow:0 0 28px rgba(255,43,214,.35); }
.card .lbl { font-size:.7rem; letter-spacing:.2em; text-transform:uppercase; color:#8fc9ff; }
.card .val { font-family:'Orbitron',sans-serif; font-size:1.9rem; font-weight:700; color:#fff; }
.card .sub { font-size:.78rem; color:#b9a7ff; }

.badge { display:inline-block; padding:6px 18px; border-radius:999px; font-weight:600;
  letter-spacing:.08em; animation: pulse 2s ease-in-out infinite; }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(255,255,255,.35)} 50%{box-shadow:0 0 0 12px rgba(255,255,255,0)} }

.stTabs [data-baseweb="tab"] { font-family:'Orbitron',sans-serif; letter-spacing:.06em; }
.stTabs [aria-selected="true"] { color:#00e5ff !important; }
h2, h3 { font-family:'Orbitron',sans-serif !important; letter-spacing:.06em; }
</style>
"""

HERO = f"""
<div class="hero">
  <h1>{APP_NAME}</h1>
  <p>{APP_TAGLINE}</p>
</div>
<div class="road"><div class="car">🏎️</div></div>
"""

GAUGE_HTML = """
<html><head><style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
body{margin:0;background:transparent;display:flex;justify-content:center;font-family:'Orbitron',sans-serif;}
.wrap{position:relative;width:340px;height:230px}
svg{width:340px;height:200px;overflow:visible}
.arc{fill:none;stroke-width:22;stroke-linecap:round}
.bg{stroke:rgba(255,255,255,.10)}
.fg{stroke:url(#g);stroke-dasharray:408.4;stroke-dashoffset:408.4;
    filter:drop-shadow(0 0 8px __COLOR__);animation:sweep 2.2s cubic-bezier(.2,.8,.2,1) forwards}
@keyframes sweep{to{stroke-dashoffset:__OFFSET__}}
.num{position:absolute;top:92px;width:100%;text-align:center;color:#fff;font-size:54px;font-weight:900;
     text-shadow:0 0 18px __COLOR__}
.unit{position:absolute;top:156px;width:100%;text-align:center;color:#9fd8ff;font-size:13px;letter-spacing:.4em}
.tick{position:absolute;color:#7fa8d6;font-size:11px}
</style></head><body>
<div class="wrap">
  <svg viewBox="0 0 300 160">
    <defs><linearGradient id="g" x1="0" x2="1">
      <stop offset="0%" stop-color="#ff2bd6"/><stop offset="50%" stop-color="#ffb300"/><stop offset="100%" stop-color="#00e5ff"/>
    </linearGradient></defs>
    <path class="arc bg" d="M 20 150 A 130 130 0 0 1 280 150"/>
    <path class="arc fg" d="M 20 150 A 130 130 0 0 1 280 150"/>
  </svg>
  <div class="tick" style="left:4px;top:176px">0</div>
  <div class="tick" style="right:0;top:176px">50+</div>
  <div class="num" id="n">0.0</div>
  <div class="unit">MPG</div>
</div>
<script>
const target=__MPG__, dur=2200, t0=performance.now();
function tick(t){const p=Math.min((t-t0)/dur,1), e=1-Math.pow(1-p,3);
  document.getElementById('n').textContent=(target*e).toFixed(1);
  if(p<1) requestAnimationFrame(tick);}
requestAnimationFrame(tick);
</script></body></html>
"""


def tier(mpg):
    if mpg >= 35:
        return "ECO COMET ☄️", "#00e5ff", "#002b33"
    if mpg >= 25:
        return "SMOOTH CRUISER 🛸", "#7CFF8A", "#06260c"
    if mpg >= 18:
        return "THIRSTY TOURER 🚀", "#ffb300", "#3a2600"
    return "GAS-GUZZLING GIANT 🦖", "#ff4d6d", "#3a0610"


def card(label, value, sub=""):
    return (f'<div class="card"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)", margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter"), legend=dict(orientation="h", y=-0.2))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6. APP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title=f"{APP_NAME} · MPG Oracle", page_icon="🏎️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(HERO, unsafe_allow_html=True)

    # --- data extraction -----------------------------------------------------
    path = find_dataset()
    raw = None
    if path is not None:
        raw = load_raw(path)
    else:
        st.warning("Dataset not found next to app.py – upload your auto-mpg CSV to begin.")
        up = st.file_uploader("Upload auto-mpg.csv", type="csv")
        if up is not None:
            raw = load_raw(up)
    if raw is None:
        st.stop()

    with st.spinner("🛰️ Calibrating the Nebula engine…"):
        art = train_everything(raw)
    df, final = art["df"], art["final"]

    # --- sidebar: the garage ---------------------------------------------------
    st.sidebar.markdown("## 🔧 THE GARAGE")
    st.sidebar.caption("Dial in your machine, the Oracle does the rest.")
    cyl_opts = sorted(int(c) for c in df["cylinders"].unique())
    cylinders = st.sidebar.select_slider("Cylinders", options=cyl_opts, value=4)
    displacement = st.sidebar.slider("Displacement (cu in)", int(df.displacement.min()),
                                     int(df.displacement.max()), 140)
    horsepower = st.sidebar.slider("Horsepower", int(df.horsepower.min()),
                                   int(df.horsepower.max()), 90)
    weight = st.sidebar.slider("Weight (lbs)", int(df.weight.min()), int(df.weight.max()), 2500, step=10)
    acceleration = st.sidebar.slider("0-60 mph time (s)", float(df.acceleration.min()),
                                     float(df.acceleration.max()), 15.5, step=0.1)
    year = st.sidebar.slider("Model year", 1900 + int(df["model year"].min()),
                             1900 + int(df["model year"].max()), 1978)
    origin = st.sidebar.selectbox("Origin", list(ORIGIN_MAP), index=2,
                                  format_func=lambda k: ORIGIN_MAP[k])

    user = pd.DataFrame([{
        "cylinders": cylinders, "displacement": displacement, "horsepower": horsepower,
        "weight": weight, "acceleration": acceleration, "model year": year - 1900,
        "origin": origin}])
    mpg = float(np.clip(final.predict(user)[0], 3, 60))
    kml, l100 = mpg * MPG_TO_KML, MPG_TO_L100 / mpg
    label, color, dark = tier(mpg)

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predict", "🧪 Model Lab", "🌌 Data Galaxy", "ℹ️ About"])

    # ============================ TAB 1: PREDICT ================================
    with tab1:
        left, right = st.columns([1.1, 1])
        with left:
            html = (GAUGE_HTML.replace("__MPG__", f"{mpg:.2f}")
                    .replace("__OFFSET__", f"{408.4 * (1 - min(mpg, 50) / 50):.1f}")
                    .replace("__COLOR__", color))
            components.html(html, height=240)
            st.markdown(
                f'<div style="text-align:center"><span class="badge" '
                f'style="background:{color};color:{dark}">{label}</span></div>',
                unsafe_allow_html=True)
        with right:
            avg = df["mpg"].mean()
            c1, c2 = st.columns(2)
            c1.markdown(card("Kilometres / litre", f"{kml:.1f}", "km per L"), unsafe_allow_html=True)
            c2.markdown(card("Litres / 100 km", f"{l100:.1f}", "lower is better"), unsafe_allow_html=True)
            st.write("")
            delta = mpg - avg
            st.markdown(card("Versus the dataset average",
                             f"{delta:+.1f} MPG",
                             f"Average car in the data: {avg:.1f} MPG"), unsafe_allow_html=True)
            st.write("")
            st.progress(min(mpg / 50, 1.0), text=f"Efficiency meter · {mpg:.1f} / 50 MPG")

        if mpg >= 35:
            st.balloons()

        st.markdown("### 🔭 What-if: weight vs efficiency")
        ws = np.linspace(df.weight.min(), df.weight.max(), 40)
        sweep = pd.concat([user] * len(ws), ignore_index=True)
        sweep["weight"] = ws
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ws, y=final.predict(sweep), mode="lines",
                                 line=dict(color="#00e5ff", width=4), name="Predicted MPG",
                                 fill="tozeroy", fillcolor="rgba(0,229,255,0.08)"))
        fig.add_trace(go.Scatter(x=[weight], y=[mpg], mode="markers", name="Your car",
                                 marker=dict(size=16, color="#ff2bd6", symbol="star",
                                             line=dict(color="white", width=1))))
        fig.update_xaxes(title="Weight (lbs)")
        fig.update_yaxes(title="MPG")
        st.plotly_chart(style_fig(fig, 340))

        st.markdown("### 🧬 Cars in the data most like yours")
        dist, idx = art["knn"].kneighbors(final[:-1].transform(user))
        similar = df.iloc[idx[0]][["car name", "model year", "cylinders", "horsepower",
                                   "weight", "mpg"]].copy()
        similar["model year"] = similar["model year"].astype(int) + 1900
        st.dataframe(similar.reset_index(drop=True), hide_index=True)

    # ============================ TAB 2: MODEL LAB ==============================
    with tab2:
        st.markdown("### 🏁 Model leaderboard")
        st.caption(f"Winner: **{art['best_name']}** (lowest 5-fold CV RMSE). "
                   "The served model is refit on 100 % of the data.")
        st.dataframe(art["board"].style.format(
            {c: "{:.3f}" for c in art["board"].columns if c != "Model"}), hide_index=True)

        a, b = st.columns(2)
        with a:
            y_te, yp = art["y_te"], art["holdout_pred"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_te, y=yp, mode="markers", name="Hold-out cars",
                                     marker=dict(color="#ff2bd6", size=8, opacity=.8)))
            lim = [min(y_te.min(), yp.min()), max(y_te.max(), yp.max())]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect",
                                     line=dict(color="#ffb300", dash="dash")))
            fig.update_layout(title="Actual vs predicted MPG (unseen test set)")
            fig.update_xaxes(title="Actual")
            fig.update_yaxes(title="Predicted")
            st.plotly_chart(style_fig(fig))
        with b:
            imp = art["importance"]
            fig = go.Figure(go.Bar(x=imp["Importance"], y=imp["Feature"], orientation="h",
                                   marker=dict(color=imp["Importance"], colorscale="Plasma")))
            fig.update_layout(title="What drives fuel economy? (permutation importance)")
            st.plotly_chart(style_fig(fig))

        st.markdown("### 🛠️ Pipeline report")
        st.markdown(
            f"- **Rows used:** {len(df)}  \n"
            f"- **Missing horsepower values imputed (median):** {art['n_missing_hp']}  \n"
            f"- **Encoding:** One-Hot on `origin`  \n"
            f"- **Scaling:** StandardScaler on {len(NUMERIC_FEATURES)} numeric features "
            f"(incl. engineered: car age, power-to-weight, displacement per cylinder)")

    # ============================ TAB 3: DATA GALAXY ============================
    with tab3:
        st.markdown("### 🌌 Every car in the data, one galaxy")
        d = df.copy()
        d["Origin"] = d["origin"].map(lambda k: ORIGIN_MAP[int(k)])
        d["Year"] = d["model year"].astype(int) + 1900
        fig = go.Figure()
        for o, g in d.groupby("Origin"):
            fig.add_trace(go.Scatter(
                x=g["weight"], y=g["mpg"], mode="markers", name=o,
                text=g["car name"], hovertemplate="%{text}<br>%{x} lbs · %{y} MPG<extra></extra>",
                marker=dict(size=g["horsepower"].fillna(g["horsepower"].median()) / 8 + 4, opacity=.7)))
        fig.add_trace(go.Scatter(x=[weight], y=[mpg], mode="markers", name="YOU",
                                 marker=dict(size=22, color="white", symbol="star",
                                             line=dict(color="#ff2bd6", width=3))))
        fig.update_xaxes(title="Weight (lbs)")
        fig.update_yaxes(title="MPG")
        st.plotly_chart(style_fig(fig, 460))

        yr = d.groupby("Year")["mpg"].mean().reset_index()
        fig2 = go.Figure(go.Scatter(x=yr["Year"], y=yr["mpg"], mode="lines+markers",
                                    line=dict(color="#7CFF8A", width=4)))
        fig2.update_layout(title="Average MPG by model year – the efficiency revolution")
        st.plotly_chart(style_fig(fig2, 320))

        with st.expander("Peek at the cleaned dataset"):
            st.dataframe(df.head(50))

    # ============================ TAB 4: ABOUT ==================================
    with tab4:
        st.markdown(f"""
### About {APP_NAME}
Predicts a car's fuel efficiency from its specs using a scikit-learn pipeline
(cleaning → feature engineering → imputation → encoding → scaling → model).

**Heads-up:** the model learned from cars built between **1970 and 1982**, so treat it as a
fun, data-driven estimate – not a prediction for modern vehicles.

**Units:** MPG is US miles per gallon. km/L and L/100 km are converted from it.
""")


if __name__ == "__main__":
    main()
