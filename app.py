"""
ipl_app.py  —  IPL Analytics Dashboard
=======================================
    streamlit run ipl_app.py
"""

import argparse, os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--out", default="data")
args, _ = parser.parse_known_args()
DATA_DIR = args.out

st.set_page_config(page_title="🏏 IPL Analytics", page_icon="🏏",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg,#1e293b,#0f172a);
    border:1px solid #334155; border-radius:12px;
    padding:20px; text-align:center; margin-bottom:8px;
}
.metric-value{font-size:2rem;font-weight:700;color:#f6c90e;}
.metric-label{font-size:.85rem;color:#94a3b8;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load(name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        st.error(f"❌ Missing: **{path}** — run `python ipl_pipeline.py --data IPL.csv --out data` first.")
        st.stop()
    return pd.read_csv(path)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🏏 IPL Analytics Dashboard")
st.markdown("PySpark · Plotly · Streamlit")
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📊 Overview",
    "🏏 Batters",
    "🎳 Bowlers",
    "🏆 Teams",
    "📅 Season Trends",
])

# ── Tab 1 · Overview ──────────────────────────────────────────────────────────
with t1:
    m = load("summary_metrics").iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [f"{int(m['matches']):,}", f"{int(m['runs']):,}",
         f"{int(m['balls']):,}", f"{int(m['wickets']):,}", str(m['avg_score'])],
        ["Matches", "Runs", "Balls", "Wickets", "Avg Score"],
    ):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                     f'<div class="metric-label">{lbl}</div></div>',
                     unsafe_allow_html=True)

    sr = load("season_runs")
    st.plotly_chart(px.bar(sr, x="year", y="total_runs", color="total_runs",
        color_continuous_scale="YlOrRd", template="plotly_dark",
        title="Total Runs Per Season"), use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        wk = load("wicket_kinds").rename(columns={"wicket_kind": "type"})
        st.plotly_chart(px.pie(wk, names="type", values="count", hole=0.4,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Wicket Types"), use_container_width=True)
    with cb:
        res = load("match_results").rename(columns={"win_outcome": "outcome"})
        st.plotly_chart(px.bar(res, x="outcome", y="count", color="outcome",
            template="plotly_dark", title="Match Results"), use_container_width=True)

# ── Tab 2 · Batters ───────────────────────────────────────────────────────────
with t2:
    n = st.slider("Top N Batters", 5, 50, 15)
    bs = load("top_batters").head(n)
    st.plotly_chart(px.bar(bs, x="batter", y="runs", color="SR",
        color_continuous_scale="RdYlGn", template="plotly_dark",
        title="Top Batters by Runs"), use_container_width=True)
    ca, cb = st.columns(2)
    with ca:
        st.plotly_chart(px.scatter(bs, x="balls", y="runs", size="SR",
            hover_name="batter", color="SR", color_continuous_scale="Plasma",
            template="plotly_dark", title="Strike Rate Bubble"),
            use_container_width=True)
    with cb:
        ov = load("runs_by_over")
        st.plotly_chart(px.area(ov, x="over", y="runs_batter",
            template="plotly_dark", color_discrete_sequence=["#f6c90e"],
            title="Runs by Over"), use_container_width=True)

# ── Tab 3 · Bowlers ───────────────────────────────────────────────────────────
with t3:
    n2 = st.slider("Top N Bowlers", 5, 50, 15)
    bw = load("top_bowlers").head(n2)
    st.plotly_chart(px.bar(bw, x="bowler", y="wickets", color="Economy",
        color_continuous_scale="RdYlGn_r", template="plotly_dark",
        title="Top Bowlers by Wickets"), use_container_width=True)
    ca, cb = st.columns(2)
    with ca:
        st.plotly_chart(px.scatter(bw, x="Economy", y="wickets",
            hover_name="bowler", size="balls", color="wickets",
            color_continuous_scale="Blues", template="plotly_dark",
            title="Economy vs Wickets"), use_container_width=True)
    with cb:
        owk = load("wickets_by_over")
        st.plotly_chart(px.bar(owk, x="over", y="wicket_kind", color="wicket_kind",
            color_continuous_scale="Reds", template="plotly_dark",
            title="Wickets by Over"), use_container_width=True)

# ── Tab 4 · Teams ─────────────────────────────────────────────────────────────
with t4:
    ts = load("team_runs")
    st.plotly_chart(px.bar(ts.sort_values("runs"), x="runs", y="batting_team",
        orientation="h", color="RPM", color_continuous_scale="YlOrRd",
        template="plotly_dark", title="Team Total Runs"), use_container_width=True)
    ca, cb = st.columns(2)
    with ca:
        td = load("toss_decisions").rename(columns={"toss_decision": "decision"})
        st.plotly_chart(px.pie(td, names="decision", values="count", hole=0.4,
            template="plotly_dark",
            color_discrete_sequence=["#f6c90e", "#3b82f6"],
            title="Toss Decisions"), use_container_width=True)
    with cb:
        tr = load("team_trends")
        st.plotly_chart(px.line(tr, x="year", y="runs_batter", color="batting_team",
            template="plotly_dark", title="Team Run Trends"),
            use_container_width=True)

# ── Tab 5 · Season Trends ─────────────────────────────────────────────────────
with t5:
    ss = load("season_stats")
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Total Runs", "Sixes & Fours", "Total Wickets", "Avg Runs/Match"])
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