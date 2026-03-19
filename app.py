"""
ipl_app.py  —  IPL Analytics Streamlit Dashboard  (with Prediction Page)
=========================================================================
Run AFTER ipl_pipeline.py has generated the data/ folder.

Usage:
    streamlit run ipl_app.py

Requirements:
    pip install -r requirements.txt
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Parse optional --out argument ─────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--out", default="data", help="Folder containing pipeline CSVs")
args, _ = parser.parse_known_args()
DATA_DIR = args.out

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏏 IPL Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body { background-color: #0f172a; }
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #f6c90e; }
.metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
.pred-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 2px solid #334155;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-top: 24px;
}
.pred-boundary    { border-color: #10b981; background: linear-gradient(135deg, #064e3b, #0f172a); }
.pred-no-boundary { border-color: #ef4444; background: linear-gradient(135deg, #450a0a, #0f172a); }
.pred-result { font-size: 2.8rem; font-weight: 800; margin-bottom: 8px; }
.pred-prob   { font-size: 1.1rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        st.error(
            f"❌  Missing file: **{path}**\n\n"
            f"Run `python ipl_pipeline.py --out {DATA_DIR}` first."
        )
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    path = os.path.join(DATA_DIR, "boundary_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def safe_encode(encoder, value: str):
    """Encode a label; fall back to index 0 for unseen values."""
    if value in list(encoder.classes_):
        return encoder.transform([value])[0]
    return 0


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏏 IPL Analytics Dashboard")
st.markdown("Powered by PySpark · scikit-learn · Plotly · Streamlit")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "📊 Overview",
    "🏏 Batters",
    "🎳 Bowlers",
    "🏆 Teams",
    "📅 Season Trends",
    "🤖 ML Results",
    "🔮 Predict Delivery",
])

# ─── Tab 1 · Overview ─────────────────────────────────────────────────────────
with t1:
    metrics = load("summary_metrics").iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [
            f"{int(metrics['matches']):,}",
            f"{int(metrics['runs']):,}",
            f"{int(metrics['balls']):,}",
            f"{int(metrics['wickets']):,}",
            str(metrics["avg_score"]),
        ],
        ["Matches", "Runs", "Balls", "Wickets", "Avg Score"],
    ):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    sr = load("season_runs")
    st.plotly_chart(
        px.bar(sr, x="year", y="total_runs", color="total_runs",
               color_continuous_scale="YlOrRd", template="plotly_dark",
               title="Total Runs Per Season"),
        use_container_width=True,
    )

    ca, cb = st.columns(2)
    with ca:
        wk = load("wicket_kinds").rename(columns={"wicket_kind": "type"})
        st.plotly_chart(
            px.pie(wk, names="type", values="count", hole=0.4,
                   template="plotly_dark",
                   color_discrete_sequence=px.colors.qualitative.Bold,
                   title="Wicket Types"),
            use_container_width=True,
        )
    with cb:
        res = load("match_results").rename(columns={"win_outcome": "outcome"})
        st.plotly_chart(
            px.bar(res, x="outcome", y="count", color="outcome",
                   template="plotly_dark", title="Match Results"),
            use_container_width=True,
        )

# ─── Tab 2 · Batters ──────────────────────────────────────────────────────────
with t2:
    n = st.slider("Top N Batters", 5, 50, 15)
    bs = load("top_batters").head(n)

    st.plotly_chart(
        px.bar(bs, x="batter", y="runs", color="SR",
               color_continuous_scale="RdYlGn", template="plotly_dark",
               title="Top Batters by Runs"),
        use_container_width=True,
    )

    ca, cb = st.columns(2)
    with ca:
        st.plotly_chart(
            px.scatter(bs, x="balls", y="runs", size="SR", hover_name="batter",
                       color="SR", color_continuous_scale="Plasma",
                       template="plotly_dark", title="Strike Rate Bubble"),
            use_container_width=True,
        )
    with cb:
        ov = load("runs_by_over")
        st.plotly_chart(
            px.area(ov, x="over", y="runs_batter", template="plotly_dark",
                    color_discrete_sequence=["#f6c90e"], title="Runs by Over"),
            use_container_width=True,
        )

# ─── Tab 3 · Bowlers ──────────────────────────────────────────────────────────
with t3:
    n2 = st.slider("Top N Bowlers", 5, 50, 15)
    bw = load("top_bowlers").head(n2)

    st.plotly_chart(
        px.bar(bw, x="bowler", y="wickets", color="Economy",
               color_continuous_scale="RdYlGn_r", template="plotly_dark",
               title="Top Bowlers by Wickets"),
        use_container_width=True,
    )

    ca, cb = st.columns(2)
    with ca:
        st.plotly_chart(
            px.scatter(bw, x="Economy", y="wickets", hover_name="bowler",
                       size="balls", color="wickets",
                       color_continuous_scale="Blues", template="plotly_dark",
                       title="Economy vs Wickets"),
            use_container_width=True,
        )
    with cb:
        owk = load("wickets_by_over")
        st.plotly_chart(
            px.bar(owk, x="over", y="wicket_kind", color="wicket_kind",
                   color_continuous_scale="Reds", template="plotly_dark",
                   title="Wickets by Over"),
            use_container_width=True,
        )

# ─── Tab 4 · Teams ────────────────────────────────────────────────────────────
with t4:
    ts = load("team_runs")
    st.plotly_chart(
        px.bar(ts.sort_values("runs"), x="runs", y="batting_team",
               orientation="h", color="RPM",
               color_continuous_scale="YlOrRd", template="plotly_dark",
               title="Team Total Runs"),
        use_container_width=True,
    )

    ca, cb = st.columns(2)
    with ca:
        td = load("toss_decisions").rename(columns={"toss_decision": "decision"})
        st.plotly_chart(
            px.pie(td, names="decision", values="count", hole=0.4,
                   template="plotly_dark",
                   color_discrete_sequence=["#f6c90e", "#3b82f6"],
                   title="Toss Decisions"),
            use_container_width=True,
        )
    with cb:
        tr = load("team_trends")
        st.plotly_chart(
            px.line(tr, x="year", y="runs_batter", color="batting_team",
                    template="plotly_dark", title="Team Run Trends by Year"),
            use_container_width=True,
        )

# ─── Tab 5 · Season Trends ────────────────────────────────────────────────────
with t5:
    ss = load("season_stats")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Total Runs", "Sixes & Fours", "Total Wickets", "Avg Runs/Match"],
    )
    fig.add_trace(go.Bar(x=ss["year"], y=ss["total_runs"],
                         marker_color="#f6c90e", name="Runs"), row=1, col=1)

    sf = load("sixes_fours")
    fig.add_trace(go.Bar(x=sf["year"], y=sf["sixes"],
                         name="Sixes", marker_color="#10b981"), row=1, col=2)
    fig.add_trace(go.Bar(x=sf["year"], y=sf["fours"],
                         name="Fours", marker_color="#3b82f6"), row=1, col=2)

    fig.add_trace(go.Scatter(x=ss["year"], y=ss["total_wickets"],
                             mode="lines+markers", line_color="#ef4444",
                             name="Wickets"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ss["year"], y=ss["avg"],
                             mode="lines+markers", line_color="#a78bfa",
                             name="Avg"), row=2, col=2)

    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

# ─── Tab 6 · ML Results ───────────────────────────────────────────────────────
with t6:
    st.markdown("### 🤖 Binary Classification — Boundary Prediction (AUC)")
    st.markdown(
        "Task: predict whether a delivery is a **boundary (4 or 6)** "
        "using over, ball, batting position, bowler, batter, and teams."
    )

    ml = load("ml_results")

    st.plotly_chart(
        px.bar(ml, x="Model", y="AUC", color="AUC",
               color_continuous_scale="RdYlGn", template="plotly_dark",
               title="Model AUC Comparison",
               text=ml["AUC"].apply(lambda x: f"{x:.4f}")),
        use_container_width=True,
    )

    ca, cb = st.columns(2)
    with ca:
        st.plotly_chart(
            px.scatter(ml, x="Time_s", y="AUC", text="Model",
                       template="plotly_dark",
                       title="AUC vs Training Time (s)",
                       color="AUC", color_continuous_scale="Blues",
                       size="AUC"),
            use_container_width=True,
        )
    with cb:
        st.dataframe(
            ml.style.format({"AUC": "{:.4f}", "Time_s": "{:.1f}s"})
              .background_gradient(subset=["AUC"], cmap="RdYlGn"),
            use_container_width=True,
        )

# ─── Tab 7 · Predict Delivery ─────────────────────────────────────────────────
with t7:
    st.markdown("### 🔮 Predict: Will This Delivery Be a Boundary?")
    st.markdown(
        "Select the match conditions below and hit **Predict** to find out "
        "if the delivery is likely to be a **four or six** — powered by a "
        "Random Forest trained on all IPL deliveries."
    )
    st.markdown("---")

    bundle = load_model()
    if bundle is None:
        st.warning(
            "⚠️  Model file not found. Re-run the pipeline:\n\n"
            f"`python ipl_pipeline.py --out {DATA_DIR}`"
        )
        st.stop()

    model    = bundle["model"]
    encoders = bundle["encoders"]

    # Load dropdown options
    teams   = load("unique_batting_team")["batting_team"].tolist()
    batters = load("unique_batter")["batter"].tolist()
    bowlers = load("unique_bowler")["bowler"].tolist()

    # ── Input form ────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### 🏏 Batting")
        batting_team = st.selectbox("Batting Team", teams, key="bat_team")
        batter       = st.selectbox("Batter", batters, key="batter")
        bat_pos      = st.number_input("Batting Position", min_value=1,
                                       max_value=11, value=1, step=1)

    with col_b:
        st.markdown("#### 🎳 Bowling")
        other_teams  = [t for t in teams if t != batting_team] or teams
        bowling_team = st.selectbox("Bowling Team", other_teams, key="bowl_team")
        bowler       = st.selectbox("Bowler", bowlers, key="bowler")

    with col_c:
        st.markdown("#### 📋 Delivery")
        over       = st.slider("Over",         min_value=0,  max_value=19, value=9)
        ball       = st.slider("Ball in Over",  min_value=1, max_value=6,  value=3)
        valid_ball = st.selectbox(
            "Valid Delivery?", [1, 0],
            format_func=lambda x: "Yes" if x else "No (Wide / No-ball)",
        )

    st.markdown("---")
    pred_col, _ = st.columns([1, 3])
    with pred_col:
        predict_clicked = st.button("🔮 Predict Delivery", use_container_width=True)

    if predict_clicked:
        # Encode inputs using fitted label encoders
        bt_enc  = safe_encode(encoders["batting_team"], batting_team)
        bwt_enc = safe_encode(encoders["bowling_team"], bowling_team)
        bat_enc = safe_encode(encoders["batter"],       batter)
        bwl_enc = safe_encode(encoders["bowler"],       bowler)

        X_input = np.array([[
            over, ball, bat_pos, valid_ball,
            bt_enc, bwt_enc, bat_enc, bwl_enc,
        ]], dtype=float)

        proba         = model.predict_proba(X_input)[0]   # [no_boundary, boundary]
        boundary_prob = float(proba[1])
        prediction    = boundary_prob >= 0.5

        # ── Result card ───────────────────────────────────────────────────────
        if prediction:
            st.markdown(
                f'<div class="pred-card pred-boundary">'
                f'<div class="pred-result">🏏 BOUNDARY! 🎉</div>'
                f'<div class="pred-prob">Probability of a boundary: '
                f'<strong>{boundary_prob:.1%}</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="pred-card pred-no-boundary">'
                f'<div class="pred-result">🛡️ No Boundary</div>'
                f'<div class="pred-prob">Probability of a boundary: '
                f'<strong>{boundary_prob:.1%}</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Probability gauge ─────────────────────────────────────────────────
        st.markdown("")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(boundary_prob * 100, 1),
            number={"suffix": "%", "font": {"color": "#f6c90e", "size": 36}},
            title={"text": "Boundary Probability", "font": {"color": "#94a3b8"}},
            gauge={
                "axis":        {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar":         {"color": "#f6c90e"},
                "bgcolor":     "#1e293b",
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0,  40],  "color": "#1e3a2f"},
                    {"range": [40, 60],  "color": "#2d3a1e"},
                    {"range": [60, 100], "color": "#3a1e1e"},
                ],
                "threshold": {
                    "line":      {"color": "#10b981", "width": 4},
                    "thickness": 0.75,
                    "value":     50,
                },
            },
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(t=40, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        _, g_mid, _ = st.columns([1, 2, 1])
        with g_mid:
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Feature importance expander ───────────────────────────────────────
        with st.expander("📊 What drove this prediction? (Feature Importances)"):
            feat_names  = ["Over", "Ball", "Bat Position", "Valid Ball",
                           "Batting Team", "Bowling Team", "Batter", "Bowler"]
            importances = model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature":    feat_names,
                "Importance": importances,
            }).sort_values("Importance", ascending=True)

            st.plotly_chart(
                px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                       color="Importance", color_continuous_scale="YlOrRd",
                       template="plotly_dark",
                       title="Random Forest Feature Importances"),
                use_container_width=True,
            )

        # ── Input summary expander ────────────────────────────────────────────
        with st.expander("🔍 Input Summary"):
            st.table(pd.DataFrame({
                "Parameter": [
                    "Batting Team", "Bowling Team", "Batter", "Bowler",
                    "Over", "Ball", "Batting Position", "Valid Delivery",
                ],
                "Value": [
                    batting_team, bowling_team, batter, bowler,
                    over, ball, bat_pos, "Yes" if valid_ball else "No",
                ],
            }))