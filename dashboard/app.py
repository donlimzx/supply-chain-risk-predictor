"""
Phase 4 — Supply Chain Risk Dashboard
======================================
Run:
    streamlit run dashboard/app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Supply Chain Risk Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared style constants ────────────────────────────────────────────────────

COLOR_LATE    = "#EF4444"   # red
COLOR_ONTIME  = "#3B82F6"   # blue
COLOR_ACCENT  = "#F59E0B"   # amber
COLOR_NEUTRAL = "#6B7280"   # gray
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_family="Inter, sans-serif",
    margin=dict(l=12, r=12, t=36, b=12),
)

# ── Data loading ──────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "DataCoSupplyChainDataset.csv"


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin-1",
                     parse_dates=["order date (DateOrders)"])
    df["order_year"] = df["order date (DateOrders)"].dt.year
    df["order_month"] = df["order date (DateOrders)"].dt.month
    return df


df_full = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📦 Supply Chain Risk")
    st.markdown("---")
    st.subheader("Filters")

    all_markets = sorted(df_full["Market"].dropna().unique())
    sel_markets = st.multiselect("Market", all_markets, default=all_markets)

    all_modes = sorted(df_full["Shipping Mode"].dropna().unique())
    sel_modes = st.multiselect("Shipping Mode", all_modes, default=all_modes)

    all_segments = sorted(df_full["Customer Segment"].dropna().unique())
    sel_segments = st.multiselect("Customer Segment", all_segments, default=all_segments)

    year_min = int(df_full["order_year"].min())
    year_max = int(df_full["order_year"].max())
    sel_years = st.slider("Order Year", year_min, year_max, (year_min, year_max))

    st.markdown("---")
    st.caption("DataCo Smart Supply Chain Dataset · Jan 2015 – Sep 2017")

# ── Apply filters ─────────────────────────────────────────────────────────────

df = df_full[
    df_full["Market"].isin(sel_markets)
    & df_full["Shipping Mode"].isin(sel_modes)
    & df_full["Customer Segment"].isin(sel_segments)
    & df_full["order_year"].between(*sel_years)
].copy()

if df.empty:
    st.warning("No data matches the current filters. Please broaden your selection.")
    st.stop()

# ── KPI calculations ──────────────────────────────────────────────────────────

total_shipments   = len(df)
pct_late          = df["Late_delivery_risk"].mean() * 100

# Most risky shipping mode = highest late rate (volume ≥ 100 to avoid noise)
mode_late = (
    df.groupby("Shipping Mode")["Late_delivery_risk"]
    .agg(["mean", "count"])
    .query("count >= 100")
    .sort_values("mean", ascending=False)
)
top_risky_mode     = mode_late.index[0] if not mode_late.empty else "N/A"
top_risky_mode_pct = mode_late["mean"].iloc[0] * 100 if not mode_late.empty else 0.0

# Top delay reason = shipping mode with the most total late shipments
mode_vol = (
    df[df["Late_delivery_risk"] == 1]
    .groupby("Shipping Mode")
    .size()
    .sort_values(ascending=False)
)
top_delay_mode     = mode_vol.index[0] if not mode_vol.empty else "N/A"
top_delay_vol      = int(mode_vol.iloc[0]) if not mode_vol.empty else 0

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("## Supply Chain Risk Dashboard")
st.markdown(
    f"Showing **{total_shipments:,}** shipments · "
    f"filtered from {len(df_full):,} total records"
)
st.markdown("---")

# ── KPI cards ─────────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)

def kpi_card(col, label: str, value: str, sub: str, color: str = COLOR_NEUTRAL) -> None:
    col.markdown(
        f"""
        <div style="
            background: #1E293B;
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 18px 20px 14px 20px;
            height: 110px;
        ">
            <div style="color:#94A3B8; font-size:12px; font-weight:600;
                        letter-spacing:0.08em; text-transform:uppercase;">{label}</div>
            <div style="color:#F1F5F9; font-size:28px; font-weight:700;
                        margin: 6px 0 4px 0; line-height:1.1;">{value}</div>
            <div style="color:#64748B; font-size:12px;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

kpi_card(k1, "Total Shipments",       f"{total_shipments:,}",
         f"{sel_years[0]}–{sel_years[1]} · filtered", COLOR_ONTIME)
kpi_card(k2, "Late Delivery Rate",    f"{pct_late:.1f}%",
         f"{int(total_shipments * pct_late / 100):,} late shipments", COLOR_LATE)
kpi_card(k3, "Top Delay Driver",      top_delay_mode,
         f"{top_delay_vol:,} late shipments via this mode", COLOR_ACCENT)
kpi_card(k4, "Most Risky Shipping Mode", top_risky_mode,
         f"{top_risky_mode_pct:.0f}% late rate", COLOR_LATE)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts: Shipping Mode + Market ────────────────────────────────────────────

col_left, col_right = st.columns(2)

# ── Left: Late rate by Shipping Mode ──────────────────────────────────────────

with col_left:
    st.subheader("Late Rate by Shipping Mode")

    mode_df = (
        df.groupby("Shipping Mode")["Late_delivery_risk"]
        .agg(late_rate="mean", count="count")
        .reset_index()
        .sort_values("late_rate", ascending=True)
    )
    mode_df["late_pct"] = mode_df["late_rate"] * 100
    mode_df["label"]    = mode_df["late_pct"].apply(lambda x: f"{x:.1f}%")

    fig_mode = go.Figure(go.Bar(
        x=mode_df["late_pct"],
        y=mode_df["Shipping Mode"],
        orientation="h",
        text=mode_df["label"],
        textposition="outside",
        marker=dict(
            color=mode_df["late_pct"],
            colorscale=[[0, COLOR_ONTIME], [0.5, COLOR_ACCENT], [1, COLOR_LATE]],
            showscale=False,
        ),
        customdata=mode_df["count"],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Late rate: %{x:.1f}%<br>"
            "Shipments: %{customdata:,}<extra></extra>"
        ),
    ))
    fig_mode.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        xaxis=dict(title="Late delivery rate (%)", gridcolor="#334155", ticksuffix="%"),
        yaxis=dict(title="", tickfont=dict(size=13)),
    )
    st.plotly_chart(fig_mode, use_container_width=True)

# ── Right: Late rate by Market ────────────────────────────────────────────────

with col_right:
    st.subheader("Late Rate by Market")

    market_df = (
        df.groupby("Market")["Late_delivery_risk"]
        .agg(late_rate="mean", count="count")
        .reset_index()
        .sort_values("late_rate", ascending=True)
    )
    market_df["late_pct"] = market_df["late_rate"] * 100
    market_df["label"]    = market_df["late_pct"].apply(lambda x: f"{x:.1f}%")

    fig_market = go.Figure(go.Bar(
        x=market_df["late_pct"],
        y=market_df["Market"],
        orientation="h",
        text=market_df["label"],
        textposition="outside",
        marker=dict(
            color=market_df["late_pct"],
            colorscale=[[0, COLOR_ONTIME], [0.5, COLOR_ACCENT], [1, COLOR_LATE]],
            showscale=False,
        ),
        customdata=market_df["count"],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Late rate: %{x:.1f}%<br>"
            "Shipments: %{customdata:,}<extra></extra>"
        ),
    ))
    fig_market.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        xaxis=dict(title="Late delivery rate (%)", gridcolor="#334155", ticksuffix="%"),
        yaxis=dict(title="", tickfont=dict(size=13)),
    )
    st.plotly_chart(fig_market, use_container_width=True)

# ── Late rate by Region (full-width) ─────────────────────────────────────────

st.subheader("Late Rate by Order Region")

region_df = (
    df.groupby("Order Region")["Late_delivery_risk"]
    .agg(late_rate="mean", count="count")
    .reset_index()
    .query("count >= 50")
    .sort_values("late_rate", ascending=True)
)
region_df["late_pct"] = region_df["late_rate"] * 100

fig_region = go.Figure(go.Bar(
    x=region_df["Order Region"],
    y=region_df["late_pct"],
    text=region_df["late_pct"].apply(lambda x: f"{x:.0f}%"),
    textposition="outside",
    marker=dict(
        color=region_df["late_pct"],
        colorscale=[[0, COLOR_ONTIME], [0.5, COLOR_ACCENT], [1, COLOR_LATE]],
        showscale=True,
        colorbar=dict(title="Late %", ticksuffix="%", thickness=14),
    ),
    customdata=region_df["count"],
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Late rate: %{y:.1f}%<br>"
        "Shipments: %{customdata:,}<extra></extra>"
    ),
))
fig_region.update_layout(
    **PLOTLY_LAYOUT,
    height=340,
    xaxis=dict(title="", tickangle=-35, gridcolor="#334155"),
    yaxis=dict(title="Late delivery rate (%)", gridcolor="#334155", ticksuffix="%"),
)
st.plotly_chart(fig_region, use_container_width=True)

# ── Data table ────────────────────────────────────────────────────────────────

st.subheader("Shipment Records")

TABLE_COLS = [
    "Order Id",
    "order date (DateOrders)",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Customer Segment",
    "Category Name",
    "Department Name",
    "Days for shipment (scheduled)",
    "Delivery Status",
    "Order Status",
    "Sales",
    "Order Profit Per Order",
    "Late_delivery_risk",
]

tc1, tc2, tc3 = st.columns([2, 2, 1])
with tc1:
    filter_risk = st.selectbox(
        "Filter by risk",
        ["All", "Late (1)", "On time / early (0)"],
    )
with tc2:
    filter_status = st.selectbox(
        "Filter by Delivery Status",
        ["All"] + sorted(df["Delivery Status"].dropna().unique()),
    )
with tc3:
    max_rows = st.selectbox("Show rows", [100, 500, 1000, 5000], index=0)

table_df = df[TABLE_COLS].copy()

if filter_risk == "Late (1)":
    table_df = table_df[table_df["Late_delivery_risk"] == 1]
elif filter_risk == "On time / early (0)":
    table_df = table_df[table_df["Late_delivery_risk"] == 0]

if filter_status != "All":
    table_df = table_df[table_df["Delivery Status"] == filter_status]

table_df = table_df.head(max_rows).reset_index(drop=True)

# Colour the risk column
def _risk_colour(val: int) -> str:
    return f"background-color: {'#7f1d1d' if val == 1 else '#14532d'}; color: white;"

st.dataframe(
    table_df.style.applymap(_risk_colour, subset=["Late_delivery_risk"]),
    use_container_width=True,
    height=420,
)

st.caption(
    f"Showing {len(table_df):,} of {len(df[TABLE_COLS]):,} filtered records. "
    "Adjust the row limit or filters above to see more."
)
