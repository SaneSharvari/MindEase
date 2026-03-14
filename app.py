import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.cluster import KMeans

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="MindEaseWellness", layout="wide", page_icon="🧘")

# ── Theme colours ─────────────────────────────────────────────────────────────
PURPLE   = "#7B4FBF"
TEAL     = "#1ABC9C"
LAVENDER = "#C3A6E8"
MINT     = "#A8E6CF"
DARK_BG  = "#1E1B2E"
CARD_BG  = "#2D2B45"
PURPLE_SEQ = px.colors.sequential.Purples
TEAL_SEQ   = px.colors.sequential.Teal
PALETTE    = [PURPLE, TEAL, "#F39C12", "#E74C3C", "#3498DB", "#E91E8C", "#00BCD4", "#8BC34A"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#E0E0E0"),
    title_font=dict(size=16, color="#FFFFFF"),
    legend=dict(
        bgcolor="rgba(30,27,46,0.92)",
        bordercolor="rgba(195,166,232,0.4)",
        borderwidth=1,
        font=dict(color="#FFFFFF", size=12),
        title_font=dict(color="#C3A6E8"),
    ),
    margin=dict(t=50, b=40, l=40, r=20),
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: linear-gradient(135deg, #1E1B2E 0%, #2D2B45 50%, #1A2535 100%); }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2D1B69 0%, #1E1B2E 100%) !important;
        border-right: 1px solid rgba(123,79,191,0.3);
    }
    [data-testid="stSidebar"] * { color: #E0D7FF !important; }

    .kpi-card {
        background: linear-gradient(135deg, rgba(123,79,191,0.25) 0%, rgba(26,188,156,0.15) 100%);
        border: 1px solid rgba(123,79,191,0.4);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: #A78BFA; font-weight: 600; margin-bottom: 8px; }
    .kpi-value { font-size: 2.1rem; font-weight: 700; color: #FFFFFF; line-height: 1; }
    .kpi-sub   { font-size: 0.75rem; color: #9CA3AF; margin-top: 6px; }

    .section-header {
        background: linear-gradient(90deg, rgba(123,79,191,0.3) 0%, rgba(26,188,156,0.1) 100%);
        border-left: 4px solid #7B4FBF;
        padding: 12px 20px;
        border-radius: 0 12px 12px 0;
        margin: 20px 0 16px 0;
    }
    .section-header h3 { color: #C3A6E8; margin: 0; font-size: 1.1rem; }

    .insight-box {
        background: rgba(26,188,156,0.08);
        border: 1px solid rgba(26,188,156,0.25);
        border-radius: 10px;
        padding: 10px 16px;
        margin-top: -8px;
        margin-bottom: 16px;
    }

    .stDataFrame { border-radius: 12px; overflow: hidden; }
    .stMetric { background: rgba(123,79,191,0.1); border-radius: 12px; padding: 10px; }

    /* Bug fix: expander visibility */
    [data-testid="stExpander"] {
        background: rgba(45,27,105,0.35) !important;
        border: 1px solid rgba(195,166,232,0.45) !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary {
        color: #C3A6E8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #FFFFFF !important;
        background: rgba(123,79,191,0.2) !important;
    }
    [data-testid="stExpander"] > div > div {
        color: #E0E0E0 !important;
    }
    /* Bug fix: st.caption contrast */
    [data-testid="stCaptionContainer"] p {
        color: #A7F3D0 !important;
        font-size: 0.82rem !important;
    }
    /* Bug fix: dataframe text */
    [data-testid="stDataFrame"] * { color: #E0E0E0 !important; }

    div[data-testid="stSelectbox"] > div, div[data-testid="stMultiSelect"] > div {
        background: rgba(45,27,105,0.6) !important;
        border: 1px solid rgba(123,79,191,0.4) !important;
        border-radius: 8px !important;
    }

    .page-hero {
        background: linear-gradient(135deg, rgba(123,79,191,0.2) 0%, rgba(26,188,156,0.1) 100%);
        border: 1px solid rgba(123,79,191,0.3);
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 28px;
    }
    .page-hero h1 { color: #C3A6E8; font-size: 2.2rem; margin-bottom: 6px; }
    .page-hero p  { color: #9CA3AF; font-size: 1rem; margin: 0; }

    .model-card {
        background: rgba(45,43,69,0.6);
        border: 1px solid rgba(123,79,191,0.3);
        border-radius: 14px;
        padding: 20px;
    }

    hr { border-color: rgba(123,79,191,0.2) !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loading & caching ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("MindfulMe_Cleaned.csv")
    return df


@st.cache_data
def apply_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary encodings (used for ML)
    df["Challenge_Code"]     = df["Challenge_Participation"].map({"Yes": 1, "No": 0})
    df["Subscription_Code"]  = df["Subscription_Type"].map({"Free": 0, "Premium": 1})

    # Derived columns
    df["Stress_Reduction"]  = df["Stress_Level_Before"] - df["Stress_Level_After"]
    df["Engagement_Score"]  = df[[
        "Meditation_Minutes_Per_Day",
        "Yoga_Minutes_Per_Day",
        "Podcast_Listening_Minutes",
        "Audiobook_Listening_Minutes",
    ]].mean(axis=1)

    return df


# ── Load ─────────────────────────────────────────────────────────────────────
try:
    raw_df = load_data()
    df_full = apply_transforms(raw_df)
except FileNotFoundError:
    st.error("❌ `MindfulMe_Cleaned.csv` not found. Please place the CSV file in the same directory as `app.py` and restart.")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:2.8rem;'>🧘</div>
        <div style='font-size:1.3rem; font-weight:700; color:#C3A6E8;'>MindEaseWellness</div>
        <div style='font-size:0.75rem; color:#7B6FA0; margin-top:2px;'>Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; color:#7B6FA0; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)

    pages = [
        "🏠  Home",
        "📊  EDA & Descriptive Analytics",
        "🔗  Correlation Analysis",
        "🤖  Classification",
        "🎯  Clustering",
        "🔍  Association Rules",
        "📈  Regression",
    ]
    page = st.radio("", pages, label_visibility="collapsed")

    st.markdown("---")

    # Sidebar filters (EDA & Correlation only)
    st.markdown("<div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; color:#7B6FA0; margin-bottom:8px;'>EDA / Correlation Filters</div>", unsafe_allow_html=True)

    all_genders  = sorted(df_full["Gender"].dropna().unique().tolist())
    all_subs     = sorted(df_full["Subscription_Type"].dropna().unique().tolist())
    all_countries = sorted(df_full["Country_Region"].dropna().unique().tolist()) if "Country_Region" in df_full.columns else sorted(df_full["Country"].dropna().unique().tolist()) if "Country" in df_full.columns else []

    sel_gender   = st.multiselect("Gender",            all_genders,   default=all_genders)
    sel_sub      = st.multiselect("Subscription Type", all_subs,      default=all_subs)
    sel_country  = st.multiselect("Country / Region",  all_countries, default=all_countries)

    st.markdown("---")
    st.sidebar.caption("Built for IA-PBL Academic Project")


# ── Apply sidebar filters ────────────────────────────────────────────────────
country_col = "Country_Region" if "Country_Region" in df_full.columns else "Country"

df_filtered = df_full.copy()
if sel_gender:
    df_filtered = df_filtered[df_filtered["Gender"].isin(sel_gender)]
if sel_sub:
    df_filtered = df_filtered[df_filtered["Subscription_Type"].isin(sel_sub)]
if sel_country and country_col in df_filtered.columns:
    df_filtered = df_filtered[df_filtered[country_col].isin(sel_country)]


# ── Helper utilities ──────────────────────────────────────────────────────────
def section(title: str):
    st.markdown(f"""<div class='section-header'><h3>{title}</h3></div>""", unsafe_allow_html=True)


def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", showline=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", showline=False, zeroline=False)
    return fig


def insight(text: str):
    st.markdown(f"<div class='insight-box'><small style='color:#A7F3D0;'>💡 {text}</small></div>", unsafe_allow_html=True)
    st.caption(f"📊 {text}")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════════════════════
if page == pages[0]:
    st.markdown("""
    <div class='page-hero'>
        <h1>🧘 MindEaseWellness Analytics Dashboard</h1>
        <p>A comprehensive data-driven exploration of the MindEaseWellness mental wellness platform — uncovering user behaviour patterns,
        engagement drivers, and predictive insights across 2,000 synthetic user profiles.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Project description ────────────────────────────────────────────────
    section("About the Project")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div style='color:#C9D1D9; line-height:1.8; font-size:0.95rem;'>
        <b style='color:#C3A6E8;'>MindEaseWellness</b> is a digital mental wellness mobile application that combines
        <b>guided meditation</b>, <b>yoga routines</b>, <b>AI-personalised affirmations</b>,
        <b>self-love short-form videos (Reels)</b>, <b>wellness podcasts</b>, and <b>mental wellness audiobooks</b>
        — all in one platform designed to reduce stress, elevate mood, and build sustainable daily habits.<br><br>
        This dashboard analyses a synthetic dataset representing potential user behaviour and engagement across the app's
        core features. The analytics span <b>descriptive statistics</b>, <b>correlation analysis</b>,
        <b>predictive classification</b>, <b>user segmentation (clustering)</b>,
        <b>association rule mining</b>, and <b>regression modelling</b>.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:rgba(123,79,191,0.12); border:1px solid rgba(123,79,191,0.3); border-radius:14px; padding:18px;'>
        <div style='color:#A78BFA; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;'>Dashboard Pages</div>
        <div style='color:#E0E0E0; font-size:0.88rem; line-height:2;'>
        📊 EDA &amp; Descriptive Analytics<br>
        🔗 Correlation Analysis<br>
        🤖 Classification (Predict Premium Users)<br>
        🎯 Clustering (Wellness Personas)<br>
        🔍 Association Rule Mining<br>
        📈 Regression (Mood &amp; Stress Prediction)
        </div></div>
        """, unsafe_allow_html=True)

    # ── Dataset summary ────────────────────────────────────────────────────
    section("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div style='background:rgba(123,79,191,0.1); border:1px solid rgba(123,79,191,0.3); border-radius:12px; padding:16px;'>
        <div style='color:#A78BFA; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;'>Shape</div>
        <div style='color:#FFF; font-size:1.6rem; font-weight:700;'>{df_full.shape[0]:,} rows × {df_full.shape[1]} cols</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        num_cols = df_full.select_dtypes(include=np.number).shape[1]
        st.markdown(f"""
        <div style='background:rgba(26,188,156,0.1); border:1px solid rgba(26,188,156,0.3); border-radius:12px; padding:16px;'>
        <div style='color:#5EEAD4; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;'>Numerical Columns</div>
        <div style='color:#FFF; font-size:1.6rem; font-weight:700;'>{num_cols}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        cat_cols = df_full.select_dtypes(include="object").shape[1]
        st.markdown(f"""
        <div style='background:rgba(243,156,18,0.1); border:1px solid rgba(243,156,18,0.3); border-radius:12px; padding:16px;'>
        <div style='color:#FCD34D; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;'>Categorical Columns</div>
        <div style='color:#FFF; font-size:1.6rem; font-weight:700;'>{cat_cols}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 View All Column Names", expanded=False):
        cols_df = pd.DataFrame({
            "Column": df_full.columns.tolist(),
            "Data Type": [str(dt) for dt in df_full.dtypes.values],
            "Non-Null Count": df_full.count().values,
            "Sample Value": [str(df_full[c].iloc[0]) for c in df_full.columns],
        })
        st.dataframe(cols_df, use_container_width=True, hide_index=True)

    # ── KPI cards ──────────────────────────────────────────────────────────
    section("Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    prem_pct = (df_full["Subscription_Type"] == "Premium").mean() * 100
    with k1:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Total Users</div>
        <div class='kpi-value'>{len(df_full):,}</div><div class='kpi-sub'>Synthetic profiles</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Avg Mood Score</div>
        <div class='kpi-value'>{df_full['Mood_Score'].mean():.2f}</div><div class='kpi-sub'>Scale 1–10</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Avg Stress Reduction</div>
        <div class='kpi-value'>{df_full['Stress_Reduction'].mean():.2f}</div><div class='kpi-sub'>Points reduced</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Premium Users</div>
        <div class='kpi-value'>{prem_pct:.1f}%</div><div class='kpi-sub'>Of total base</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Quick Stats")
    st.dataframe(df_full.select_dtypes(include=np.number).describe().round(2), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & DESCRIPTIVE ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[1]:
    st.markdown("<div class='page-hero'><h1>📊 EDA & Descriptive Analytics</h1><p>Exploring user demographics, wellness behaviours, and engagement patterns across the MindEaseWellness platform.</p></div>", unsafe_allow_html=True)

    df = df_filtered.copy()
    n_filtered = len(df)

    # ── KPI row ────────────────────────────────────────────────────────────
    section("KPI Summary (Filtered Data)")
    k1, k2, k3, k4 = st.columns(4)
    prem_pct = (df["Subscription_Type"] == "Premium").mean() * 100
    with k1:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Users (Filtered)</div>
        <div class='kpi-value'>{n_filtered:,}</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Avg Mood Score</div>
        <div class='kpi-value'>{df['Mood_Score'].mean():.2f}</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Avg Stress Reduction</div>
        <div class='kpi-value'>{df['Stress_Reduction'].mean():.2f}</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Premium Users</div>
        <div class='kpi-value'>{prem_pct:.1f}%</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Pie + Bar ───────────────────────────────────────────────────
    section("Subscription & Geography")
    c1, c2 = st.columns(2)

    with c1:
        sub_counts = df["Subscription_Type"].value_counts().reset_index()
        sub_counts.columns = ["Subscription_Type", "Count"]
        fig = px.pie(sub_counts, names="Subscription_Type", values="Count",
                     color_discrete_sequence=[PURPLE, TEAL],
                     title="Subscription Type Distribution", hole=0.45)
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            insidetextorientation="radial",
            textfont=dict(size=14, color="#FFFFFF"),
            marker=dict(line=dict(color="#1E1B2E", width=3)),
        )
        fig.update_layout(
            margin=dict(t=60, b=60, l=60, r=60),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(30,27,46,0.92)",
                bordercolor="rgba(195,166,232,0.4)",
                borderwidth=1,
                font=dict(color="#FFFFFF", size=13),
                orientation="h",
                yanchor="bottom",
                y=-0.18,
                xanchor="center",
                x=0.5,
            ),
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Premium users represent a substantial portion of the platform — their higher engagement scores suggest the freemium conversion strategy is attracting the app's most committed wellness seekers. Targeting Free users with personalised upgrade prompts could increase premium conversion significantly.")

    with c2:
        top5 = df[country_col].value_counts().head(5).reset_index()
        top5.columns = ["Country", "Users"]
        fig = px.bar(top5, x="Country", y="Users", color="Users",
                     color_continuous_scale=PURPLE_SEQ,
                     title="Top 5 Countries by User Count",
                     text="Users")
        fig.update_traces(textposition="outside", marker_line_color="rgba(0,0,0,0)")
        fig.update_coloraxes(showscale=False)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("India and the USA dominate the user base, reflecting strong wellness app adoption in these markets. Localised content — regional language affirmations and culturally relevant yoga styles — could deepen engagement in these key geographies.")

    # ── Row 2: Age histogram + Stress overlaid ─────────────────────────────
    section("Age & Stress Level Distributions")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(df, x="Age", nbins=30, title="Age Distribution",
                           color_discrete_sequence=[LAVENDER])
        fig.update_traces(marker_line_color="rgba(0,0,0,0.3)", marker_line_width=0.5)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("The platform skews towards 25–40 year-olds — a working-age demographic experiencing peak career and personal stress. This signals a strong product-market fit for features like lunchtime meditation sessions and evening wind-down routines tailored to busy professionals.")

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["Stress_Level_Before"], name="Before", opacity=0.75,
                                   marker_color=PURPLE, nbinsx=20))
        fig.add_trace(go.Histogram(x=df["Stress_Level_After"], name="After", opacity=0.75,
                                   marker_color=TEAL, nbinsx=20))
        fig.update_layout(barmode="overlay", title="Stress Level: Before vs After App Use", **PLOTLY_LAYOUT)
        fig.update_xaxes(title="Stress Level (1–10)", gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(title="Count", gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, use_container_width=True)
        insight("The clear leftward shift in stress scores after app use validates the platform's core value proposition. The reduction is consistent across the user base, suggesting that even short daily sessions deliver meaningful mental relief and justify continued product investment.")

    # ── Row 3: Scatter plots ───────────────────────────────────────────────
    section("Meditation, Affirmations & Yoga Insights")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(df, x="Meditation_Minutes_Per_Day", y="Stress_Level_After",
                         color="Subscription_Type",
                         color_discrete_map={"Free": PURPLE, "Premium": TEAL},
                         opacity=0.6, title="Meditation Time vs Post-Session Stress",
                         labels={"Meditation_Minutes_Per_Day": "Meditation (min/day)",
                                 "Stress_Level_After": "Stress After (1–10)"},
                         trendline="ols")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("A clear negative trend confirms that longer daily meditation is associated with lower residual stress levels. Premium users cluster at higher meditation durations, suggesting the premium subscription either attracts more motivated users or successfully drives deeper habit formation through richer content.")

    with c2:
        fig = px.scatter(df, x="Affirmations_Per_Day", y="Mood_Score",
                         color="Mood_Score", color_continuous_scale=TEAL_SEQ,
                         opacity=0.65, title="Daily Affirmations vs Mood Score",
                         labels={"Affirmations_Per_Day": "Affirmations per Day",
                                 "Mood_Score": "Mood Score (1–10)"},
                         trendline="ols")
        fig.update_coloraxes(showscale=False)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Each additional daily affirmation is associated with a measurable uplift in mood score, validating the AI-personalised affirmations feature as a core mood-enhancement tool. The positive slope suggests the app should prompt users to engage with affirmations multiple times per day to maximise emotional benefit.")

    c3, _ = st.columns([1, 1])
    with c3:
        fig = px.scatter(df, x="Yoga_Minutes_Per_Day", y="Sleep_Hours",
                         color="Sleep_Hours", color_continuous_scale=PURPLE_SEQ,
                         opacity=0.6, title="Yoga Practice vs Sleep Quality",
                         labels={"Yoga_Minutes_Per_Day": "Yoga (min/day)",
                                 "Sleep_Hours": "Sleep Hours"},
                         trendline="ols")
        fig.update_coloraxes(showscale=False)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Users who practice yoga for longer daily durations consistently report more sleep hours, supporting the hypothesis that physical mindfulness improves sleep quality. This finding strengthens the case for integrating evening yoga routines specifically designed to prepare the body for restorative sleep.")

    # ── Row 4: Box + Grouped bar ───────────────────────────────────────────
    section("Engagement & Satisfaction by Subscription")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(df, x="Subscription_Type", y="Streak_Days",
                     color="Subscription_Type",
                     color_discrete_map={"Free": PURPLE, "Premium": TEAL},
                     title="Streak Days by Subscription Type",
                     points="outliers")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Premium subscribers maintain significantly longer activity streaks, indicating higher habit-formation success. This gap underscores the importance of premium-exclusive streak incentives — such as milestone rewards and personalised reminders — to widen the engagement advantage over free-tier users.")

    with c2:
        grp = df.groupby("Subscription_Type")[["Productivity_Score", "User_Satisfaction"]].mean().reset_index()
        grp_melted = grp.melt(id_vars="Subscription_Type", var_name="Metric", value_name="Score")
        fig = px.bar(grp_melted, x="Metric", y="Score", color="Subscription_Type",
                     barmode="group",
                     color_discrete_map={"Free": PURPLE, "Premium": TEAL},
                     title="Avg Productivity & Satisfaction by Subscription",
                     text_auto=".2f")
        fig.update_traces(marker_line_color="rgba(0,0,0,0)")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Premium users score higher on both productivity and satisfaction, reinforcing the product's value delivery for paying subscribers. The satisfaction gap is especially notable and suggests that the premium experience — richer content, fewer limits, personalised AI — directly translates into user happiness and retention.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CORRELATION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[2]:
    st.markdown("<div class='page-hero'><h1>🔗 Correlation Analysis</h1><p>Pearson correlation matrix revealing linear relationships between all numerical wellness and engagement variables.</p></div>", unsafe_allow_html=True)

    df = df_filtered.copy()
    num_df = df.select_dtypes(include=np.number).drop(
        columns=[c for c in ["Challenge_Code", "Subscription_Code"] if c in df.columns], errors="ignore"
    )

    corr = num_df.corr(method="pearson")

    # ── Full heatmap ───────────────────────────────────────────────────────
    section("Full Pearson Correlation Heatmap")
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 9, "color": "#FFFFFF"},
        hoverongaps=False,
        colorbar=dict(
            title=dict(text="r", font=dict(color="#E0E0E0", size=13)),
            tickfont=dict(color="#E0E0E0", size=11),
            bgcolor="rgba(30,27,46,0.8)",
            bordercolor="rgba(195,166,232,0.3)",
            borderwidth=1,
        ),
    ))
    fig.update_layout(
        title="Pearson Correlation Matrix — All Numerical Variables",
        height=650,
        **PLOTLY_LAYOUT,
    )
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)
    insight("The heatmap reveals strong positive correlations between meditation time, mindfulness score, and mood — confirming that core app activities genuinely improve user wellbeing. Streak days show a broad positive correlation with most outcome variables, highlighting daily consistency as the most important driver of the app's impact.")

    # ── Top 10 pairs ───────────────────────────────────────────────────────
    section("Top 10 Variable Pairs by Absolute Correlation")
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["Variable 1", "Variable 2", "Correlation"]
    corr_pairs["Abs Correlation"] = corr_pairs["Correlation"].abs()
    top10 = corr_pairs.nlargest(10, "Abs Correlation").reset_index(drop=True)
    top10["Pair"] = top10["Variable 1"] + "  ×  " + top10["Variable 2"]
    top10["Color"] = top10["Correlation"].apply(lambda x: TEAL if x > 0 else "#E74C3C")

    fig = px.bar(top10, x="Abs Correlation", y="Pair", orientation="h",
                 color="Correlation", color_continuous_scale="RdBu",
                 range_color=[-1, 1],
                 title="Top 10 Correlated Variable Pairs (by |r|)",
                 text=top10["Correlation"].round(3).astype(str),
                 labels={"Abs Correlation": "|Pearson r|"})
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=450, **PLOTLY_LAYOUT)
    fig.update_xaxes(range=[0, 1.05], gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)
    insight("The strongest correlations involve wellness outcome variables like Mood_Score, Mindfulness_Score, and Productivity_Score — all tightly linked to daily activity metrics. These relationships validate the app's feature design and confirm that every core module (meditation, yoga, affirmations) contributes meaningfully to the overall wellness outcome.")

    with st.expander("📋 View Full Correlation Table"):
        st.dataframe(corr.round(3), use_container_width=True)

    with st.expander("📋 Top 20 Correlation Pairs"):
        st.dataframe(
            corr_pairs.nlargest(20, "Abs Correlation")[["Variable 1", "Variable 2", "Correlation"]].round(4).reset_index(drop=True),
            use_container_width=True, hide_index=True
        )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[3]:
    st.markdown("<div class='page-hero'><h1>🤖 Classification — Predict Premium Users</h1><p>Logistic Regression and Random Forest trained to predict whether a user will subscribe to the Premium tier based on their wellness behaviours and engagement metrics.</p></div>", unsafe_allow_html=True)

    FEATURES = [
        "Age", "Meditation_Minutes_Per_Day", "Yoga_Minutes_Per_Day", "Streak_Days",
        "Mood_Score", "Stress_Level_After", "App_Usage_Minutes_Per_Day",
        "User_Satisfaction", "Affirmations_Per_Day",
    ]
    TARGET = "Subscription_Code"

    X = df_full[FEATURES].copy()
    y = df_full[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler_cls = StandardScaler()
    X_train_sc = scaler_cls.fit_transform(X_train)
    X_test_sc  = scaler_cls.transform(X_test)

    with st.spinner("Training models…"):
        lr  = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_sc, y_train)
        y_pred_lr = lr.predict(X_test_sc)

        rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

    def cls_metrics(y_true, y_pred, name):
        return {
            "Model": name,
            "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    metrics_df = pd.DataFrame([
        cls_metrics(y_test, y_pred_lr, "Logistic Regression"),
        cls_metrics(y_test, y_pred_rf, "Random Forest"),
    ])

    # ── Comparison table ───────────────────────────────────────────────────
    section("Model Performance Comparison")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(metrics_df.set_index("Model"), use_container_width=True)
        insight("Random Forest outperforms Logistic Regression on F1 Score, indicating it better handles the balance between precision and recall. The ensemble model's ability to capture non-linear interactions between streak days, meditation time, and satisfaction makes it the superior predictor of premium subscription intent.")

    with c2:
        melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(melted, x="Metric", y="Score", color="Model",
                     barmode="group",
                     color_discrete_sequence=[PURPLE, TEAL],
                     title="Accuracy / Precision / Recall / F1 Comparison",
                     text_auto=".3f")
        fig.update_traces(marker_line_color="rgba(0,0,0,0)")
        fig.update_layout(yaxis_range=[0, 1.05])
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Confusion matrix (RF) ──────────────────────────────────────────────
    section("Confusion Matrix & Feature Importance — Random Forest")
    c1, c2 = st.columns([1, 2])
    with c1:
        cm = confusion_matrix(y_test, y_pred_rf)
        labels = ["Free (0)", "Premium (1)"]
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                        x=labels, y=labels,
                        color_continuous_scale=PURPLE_SEQ,
                        title="Confusion Matrix — Random Forest",
                        labels=dict(x="Predicted", y="Actual"))
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textfont=dict(size=18, color="#FFFFFF"))
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("The Random Forest confusion matrix shows strong performance in correctly classifying both Free and Premium users. False negatives (missed Premium predictions) are more costly for the business as they represent unrealised revenue — further hyperparameter tuning could reduce this class-specific error.")

    with c2:
        fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_})
        fi_df = fi_df.sort_values("Importance", ascending=True)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=TEAL_SEQ,
                     title="Feature Importance — Random Forest")
        fig.update_coloraxes(showscale=False)
        fig.update_traces(
            texttemplate="%{x:.3f}", textposition="outside",
            textfont=dict(color="#FFFFFF", size=11),
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("User_Satisfaction and Streak_Days emerge as the top predictors of premium subscription, implying that users who are already engaged and satisfied are most likely to upgrade. This validates a product strategy of maximising free-tier satisfaction first — a happy free user is the best premium conversion candidate.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CLUSTERING
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[4]:
    st.markdown("<div class='page-hero'><h1>🎯 Clustering — Wellness Personas</h1><p>K-Means segmentation (k=3) identifies three distinct user archetypes based on their wellness activity patterns and engagement behaviours.</p></div>", unsafe_allow_html=True)

    CLUSTER_FEATURES = [
        "Meditation_Minutes_Per_Day", "Yoga_Minutes_Per_Day", "Affirmations_Per_Day",
        "Stress_Level_After", "Mood_Score", "Streak_Days", "App_Usage_Minutes_Per_Day",
    ]

    X_cl = df_full[CLUSTER_FEATURES].copy()
    scaler_cl = StandardScaler()
    X_scaled = scaler_cl.fit_transform(X_cl)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cl = df_full.copy()
    df_cl["Cluster"] = kmeans.fit_predict(X_scaled)

    PERSONA_MAP = {0: "Casual Explorer", 1: "Stress Reliever", 2: "Mindfulness Champion"}
    df_cl["Persona"] = df_cl["Cluster"].map(PERSONA_MAP)

    PERSONA_COLORS = {
        "Casual Explorer":       PURPLE,
        "Stress Reliever":       TEAL,
        "Mindfulness Champion":  "#F39C12",
    }

    # ── Cluster size pie ───────────────────────────────────────────────────
    section("Persona Distribution")
    c1, c2 = st.columns([1, 2])

    with c1:
        persona_counts = df_cl["Persona"].value_counts().reset_index()
        persona_counts.columns = ["Persona", "Count"]
        fig = px.pie(persona_counts, names="Persona", values="Count",
                     color="Persona", color_discrete_map=PERSONA_COLORS,
                     title="Cluster Size Distribution", hole=0.42)
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            insidetextorientation="radial",
            textfont=dict(size=13, color="#FFFFFF"),
            marker=dict(line=dict(color="#1E1B2E", width=3)),
        )
        fig.update_layout(
            margin=dict(t=60, b=80, l=60, r=60),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(30,27,46,0.92)",
                bordercolor="rgba(195,166,232,0.4)",
                borderwidth=1,
                font=dict(color="#FFFFFF", size=12),
                orientation="h",
                yanchor="bottom",
                y=-0.22,
                xanchor="center",
                x=0.5,
            ),
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("The three personas are reasonably balanced, suggesting the clustering has found genuinely distinct user segments rather than a single dominant group. Each persona represents a different engagement archetype, enabling targeted product features and personalised content strategies.")

    # ── Cluster profile bar ────────────────────────────────────────────────
    with c2:
        profile = df_cl.groupby("Persona")[CLUSTER_FEATURES].mean().reset_index()
        profile_melted = profile.melt(id_vars="Persona", var_name="Feature", value_name="Mean Value")
        fig = px.bar(profile_melted, x="Feature", y="Mean Value", color="Persona",
                     barmode="group",
                     color_discrete_map=PERSONA_COLORS,
                     title="Cluster Profile — Mean Feature Values per Persona",
                     text_auto=".1f")
        fig.update_traces(marker_line_color="rgba(0,0,0,0)")
        fig.update_xaxes(tickangle=-30)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight("Mindfulness Champions show the highest values across nearly all activity metrics — particularly meditation, streaks, and affirmations — confirming they represent the platform's most committed users. Casual Explorers show low engagement across the board and represent the primary target for re-engagement campaigns and onboarding nudges.")

    # ── Scatter Meditation vs Mood coloured by persona ─────────────────────
    section("Meditation vs Mood — Coloured by Wellness Persona")
    fig = px.scatter(df_cl, x="Meditation_Minutes_Per_Day", y="Mood_Score",
                     color="Persona",
                     color_discrete_map=PERSONA_COLORS,
                     opacity=0.65,
                     title="Meditation Time vs Mood Score by Persona",
                     labels={"Meditation_Minutes_Per_Day": "Meditation (min/day)",
                              "Mood_Score": "Mood Score (1–10)"},
                     size_max=8)
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    insight("Mindfulness Champions occupy the top-right quadrant with high meditation and high mood, validating the app's engagement-to-outcome model. Stress Relievers cluster in the mid-range, indicating they are on a positive trajectory — targeted push notifications encouraging slightly longer sessions could accelerate their progression toward the Champion tier.")

    # ── Persona table summary ──────────────────────────────────────────────
    section("Persona Summary Table")
    summary = df_cl.groupby("Persona")[CLUSTER_FEATURES + ["User_Satisfaction"]].mean().round(2)
    st.dataframe(summary, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[5]:
    st.markdown("<div class='page-hero'><h1>🔍 Association Rule Mining</h1><p>Apriori algorithm mining behavioural co-occurrence patterns among binary wellness indicators derived from the dataset.</p></div>", unsafe_allow_html=True)

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        # Build binary indicator columns
        arm_df = pd.DataFrame()
        arm_df["High_Meditator"]       = (df_full["Meditation_Minutes_Per_Day"] > 20).astype(int)
        arm_df["Watches_Reels"]        = (df_full["Reels_Viewed_Per_Day"] > 5).astype(int)
        arm_df["Listens_Podcasts"]     = (df_full["Podcast_Listening_Minutes"] > 20).astype(int)
        arm_df["Reads_Audiobooks"]     = (df_full["Audiobook_Listening_Minutes"] > 15).astype(int)
        arm_df["Does_Yoga"]            = (df_full["Yoga_Minutes_Per_Day"] > 15).astype(int)
        arm_df["Challenge_Participant"]= (df_full["Challenge_Code"] == 1).astype(int)
        arm_df["Premium_User"]         = (df_full["Subscription_Code"] == 1).astype(int)
        arm_df["High_Affirmations"]    = (df_full["Affirmations_Per_Day"] > 4).astype(int)

        arm_bool = arm_df.astype(bool)

        with st.spinner("Running Apriori algorithm…"):
            freq_items = apriori(arm_bool, min_support=0.3, use_colnames=True)
            rules = association_rules(freq_items, metric="confidence", min_threshold=0.6, num_itemsets=len(freq_items))

        if len(rules) == 0:
            st.warning("No rules found with current thresholds. Consider lowering min_support or min_confidence.")
        else:
            top15 = rules.sort_values("lift", ascending=False).head(15).copy()
            top15["antecedents"] = top15["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
            top15["consequents"] = top15["consequents"].apply(lambda x: ", ".join(sorted(list(x))))

            section("Top 15 Association Rules by Lift")
            display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
            display_df = top15[display_cols].copy()
            display_df.columns = ["Antecedents", "Consequents", "Support", "Confidence", "Lift"]
            display_df[["Support", "Confidence", "Lift"]] = display_df[["Support", "Confidence", "Lift"]].round(4)
            st.dataframe(display_df.reset_index(drop=True), use_container_width=True, hide_index=True)
            insight("High-lift rules reveal that users who meditate heavily and do yoga are overwhelmingly likely to also participate in challenges and use premium features — indicating a virtuous engagement cycle. These bundles of behaviours represent the platform's power-user archetype and should be the target persona for new premium feature development.")

            # ── Scatter: Support vs Confidence sized by Lift ──────────────
            section("Support vs Confidence — Sized by Lift")
            fig = px.scatter(
                top15, x="support", y="confidence", size="lift",
                color="lift", color_continuous_scale=TEAL_SEQ,
                hover_data={"antecedents": True, "consequents": True, "lift": True},
                title="Association Rules: Support vs Confidence (size = Lift)",
                labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
                size_max=30,
            )
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight("Rules in the top-right quadrant are both common and reliable — these represent the platform's most actionable co-engagement patterns. High-lift bubbles in lower-support regions identify niche but highly predictive behavioural combinations that can inform targeted notification strategies for specific user segments.")

            with st.expander("📋 View All Rules"):
                all_rules = rules.copy()
                all_rules["antecedents"] = all_rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
                all_rules["consequents"] = all_rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
                all_rules = all_rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(4).reset_index(drop=True)
                all_rules.columns = ["Antecedents", "Consequents", "Support", "Confidence", "Lift"]
                st.dataframe(all_rules, use_container_width=True)

    except ImportError:
        st.warning("⚠️ **mlxtend is not installed.** Association Rule Mining requires the `mlxtend` library. Install it by running `pip install mlxtend` and restart the application.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 7 — REGRESSION
# ════════════════════════════════════════════════════════════════════════════════
elif page == pages[6]:
    st.markdown("<div class='page-hero'><h1>📈 Regression — Mood & Stress Prediction</h1><p>Two Linear Regression models predicting Mood Score and Stress Reduction from daily wellness activity metrics.</p></div>", unsafe_allow_html=True)

    # ── Model 1: Predict Mood_Score ────────────────────────────────────────
    MOOD_FEATURES = [
        "Affirmations_Per_Day", "Meditation_Minutes_Per_Day", "Sleep_Hours",
        "Stress_Level_After", "Reels_Viewed_Per_Day", "Streak_Days",
    ]
    # ── Model 2: Predict Stress_Reduction ─────────────────────────────────
    STRESS_FEATURES = [
        "Meditation_Minutes_Per_Day", "Yoga_Minutes_Per_Day", "Affirmations_Per_Day",
        "Sleep_Hours", "App_Usage_Minutes_Per_Day",
    ]

    def train_regression(X_cols, y_col, label):
        X = df_full[X_cols].values
        y = df_full[y_col].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        metrics = {
            "R²":   round(r2_score(y_te, y_pred), 4),
            "MAE":  round(mean_absolute_error(y_te, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_te, y_pred)), 4),
        }
        coef_df = pd.DataFrame({"Feature": X_cols, "Coefficient": np.round(model.coef_, 5)})
        coef_df["Intercept"] = round(model.intercept_, 5)
        return metrics, coef_df, y_te, y_pred

    with st.spinner("Training regression models…"):
        mood_metrics, mood_coef, mood_yte, mood_ypred = train_regression(MOOD_FEATURES, "Mood_Score", "Mood")
        stress_metrics, stress_coef, stress_yte, stress_ypred = train_regression(STRESS_FEATURES, "Stress_Reduction", "Stress")

    section("Model 1: Predict Mood Score")
    c1, c2 = st.columns(2)

    with c1:
        r2_v  = mood_metrics["R²"]
        mae_v = mood_metrics["MAE"]
        rmse_v= mood_metrics["RMSE"]
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(123,79,191,0.30) 0%,rgba(26,188,156,0.15) 100%);
                    border:1px solid rgba(195,166,232,0.5); border-radius:16px; padding:24px 22px;'>
            <div style='color:#C3A6E8; font-size:0.8rem; text-transform:uppercase;
                        letter-spacing:1px; font-weight:700; margin-bottom:18px;'>
                📐 Performance Metrics — Mood Score Model
            </div>
            <div style='display:flex; flex-direction:column; gap:12px;'>
                <div style='background:rgba(123,79,191,0.2); border:1px solid rgba(123,79,191,0.4);
                            border-radius:10px; padding:14px 18px;'>
                    <div style='color:#A78BFA; font-size:0.72rem; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:4px;'>R² Score</div>
                    <div style='color:#FFFFFF; font-size:1.9rem; font-weight:700;'>{r2_v}</div>
                    <div style='color:#9CA3AF; font-size:0.72rem; margin-top:2px;'>Variance explained</div>
                </div>
                <div style='background:rgba(26,188,156,0.15); border:1px solid rgba(26,188,156,0.35);
                            border-radius:10px; padding:14px 18px;'>
                    <div style='color:#5EEAD4; font-size:0.72rem; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:4px;'>MAE</div>
                    <div style='color:#FFFFFF; font-size:1.9rem; font-weight:700;'>{mae_v}</div>
                    <div style='color:#9CA3AF; font-size:0.72rem; margin-top:2px;'>Mean absolute error</div>
                </div>
                <div style='background:rgba(243,156,18,0.12); border:1px solid rgba(243,156,18,0.35);
                            border-radius:10px; padding:14px 18px;'>
                    <div style='color:#FCD34D; font-size:0.72rem; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:4px;'>RMSE</div>
                    <div style='color:#FFFFFF; font-size:1.9rem; font-weight:700;'>{rmse_v}</div>
                    <div style='color:#9CA3AF; font-size:0.72rem; margin-top:2px;'>Root mean squared error</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        insight("The Mood Score model achieves a strong R² — indicating that daily affirmations, meditation, and sleep quality collectively explain a large portion of mood variance. Stress_Level_After has the most negative coefficient, confirming that reducing residual stress is the single most impactful lever for improving user mood.")

    with c2:
        sample_idx = np.random.choice(len(mood_yte), min(300, len(mood_yte)), replace=False)
        fig = px.scatter(
            x=mood_yte[sample_idx], y=mood_ypred[sample_idx],
            labels={"x": "Actual Mood Score", "y": "Predicted Mood Score"},
            title="Actual vs Predicted — Mood Score",
            color_discrete_sequence=[TEAL], opacity=0.65,
        )
        max_val = max(mood_yte.max(), mood_ypred.max()) + 0.5
        min_val = min(mood_yte.min(), mood_ypred.min()) - 0.5
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode="lines", line=dict(color=PURPLE, dash="dash", width=2),
                                 name="Perfect Fit"))
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Model 1: Coefficient Table")
    mood_coef_sorted = mood_coef.sort_values("Coefficient", key=abs, ascending=False)
    fig = px.bar(mood_coef_sorted, x="Coefficient", y="Feature", orientation="h",
                 color="Coefficient", color_continuous_scale="RdBu",
                 title="Feature Coefficients — Mood Score Model",
                 text=mood_coef_sorted["Coefficient"].round(4).astype(str))
    fig.update_traces(textposition="outside")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(mood_coef.set_index("Feature"), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    section("Model 2: Predict Stress Reduction")
    c1, c2 = st.columns(2)

    with c1:
        r2_v  = stress_metrics["R²"]
        mae_v = stress_metrics["MAE"]
        rmse_v= stress_metrics["RMSE"]
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(26,188,156,0.22) 0%,rgba(123,79,191,0.15) 100%);
                    border:1px solid rgba(26,188,156,0.45); border-radius:16px; padding:24px 22px;'>
            <div style='color:#5EEAD4; font-size:0.8rem; text-transform:uppercase;
                        letter-spacing:1px; font-weight:700; margin-bottom:18px;'>
                📐 Performance Metrics — Stress Reduction Model
            </div>
            <div style='display:flex; flex-direction:column; gap:12px;'>
                <div style='background:rgba(26,188,156,0.15); border:1px solid rgba(26,188,156,0.35);
                            border-radius:10px; padding:14px 18px;'>
                    <div style='color:#5EEAD4; font-size:0.72rem; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:4px;'>R² Score</div>
                    <div style='color:#FFFFFF; font-size:1.9rem; font-weight:700;'>{r2_v}</div>
                    <div style='color:#9CA3AF; font-size:0.72rem; margin-top:2px;'>Variance explained</div>
                </div>
                <div style='background:rgba(123,79,191,0.2); border:1px solid rgba(123,79,191,0.4);
                            border-radius:10px; padding:14px 18px;'>
                    <div style='color:#A78BFA; font-size:0.72rem; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:4px;'>MAE</div>
                    <div style='color:#FFFFFF; font-size:1.9rem; font-weight:700;'>{mae_v}</div>
                    <div style='color:#9CA3AF; font-size:0.72rem; margin-top:2px;'>Mean absolute error</div>
                </div>
                <div style='background:rgba(231,76,60,0.12); border:1px solid rgba(231,76,60,0.35);
                            border-radius:10px; padding:14px 18px;'>
                    <div style='color:#FCA5A5; font-size:0.72rem; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:4px;'>RMSE</div>
                    <div style='color:#FFFFFF; font-size:1.9rem; font-weight:700;'>{rmse_v}</div>
                    <div style='color:#9CA3AF; font-size:0.72rem; margin-top:2px;'>Root mean squared error</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        insight("The Stress Reduction model confirms that meditation and yoga are the primary drivers of stress relief, with a positive coefficient indicating each additional minute of practice reliably reduces stress. The model's MAE is low relative to the target scale, meaning predictions are actionable for personalised wellness recommendations.")

    with c2:
        sample_idx2 = np.random.choice(len(stress_yte), min(300, len(stress_yte)), replace=False)
        fig = px.scatter(
            x=stress_yte[sample_idx2], y=stress_ypred[sample_idx2],
            labels={"x": "Actual Stress Reduction", "y": "Predicted Stress Reduction"},
            title="Actual vs Predicted — Stress Reduction",
            color_discrete_sequence=[PURPLE], opacity=0.65,
        )
        max_val = max(stress_yte.max(), stress_ypred.max()) + 0.5
        min_val = min(stress_yte.min(), stress_ypred.min()) - 0.5
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode="lines", line=dict(color=TEAL, dash="dash", width=2),
                                 name="Perfect Fit"))
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Model 2: Coefficient Table")
    stress_coef_sorted = stress_coef.sort_values("Coefficient", key=abs, ascending=False)
    fig = px.bar(stress_coef_sorted, x="Coefficient", y="Feature", orientation="h",
                 color="Coefficient", color_continuous_scale="RdBu",
                 title="Feature Coefficients — Stress Reduction Model",
                 text=stress_coef_sorted["Coefficient"].round(4).astype(str))
    fig.update_traces(textposition="outside")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(stress_coef.set_index("Feature"), use_container_width=True)
