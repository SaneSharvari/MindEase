# 🧘 MindfulMe Analytics Dashboard

A comprehensive multi-page **Streamlit analytics dashboard** built for the **IA-PBL Academic Project**, analysing 2,000 synthetic user profiles from the **MindfulMe** mental wellness mobile application.

---

## 📌 Project Overview

**MindfulMe** is a digital mental wellness platform combining:
- 🧘 Guided Meditation & Yoga Routines
- 💬 AI-Personalised Daily Affirmations
- 🎬 Self-Love Reels (short-form video feed)
- 🎙️ Wellness Podcasts & Audiobooks
- 📊 Mood & Stress Tracking Dashboard
- 🏆 Daily Streak Tracker & Wellness Challenges

This dashboard analyses user behaviour, validates the business concept with data, and builds predictive models to support decision-making.

---

## 🗂️ Repository Structure

```
mindfulme-dashboard/
│
├── app.py                        # Main Streamlit application (single file)
├── MindfulMe_Cleaned.csv         # Input dataset (2,000 rows × 22 columns)
├── requirements.txt              # Python package dependencies
└── README.md                     # This file
```

---

## 📊 Dashboard Pages

| # | Page | Description |
|---|------|-------------|
| 1 | 🏠 **Home** | Project overview, dataset summary, 4 KPI cards |
| 2 | 📊 **EDA & Descriptive Analytics** | 4 KPIs, 7+ Plotly charts, sidebar filters |
| 3 | 🔗 **Correlation Analysis** | Full Pearson heatmap + top-10 pairs bar chart |
| 4 | 🤖 **Classification** | Predict Premium users — Logistic Regression vs Random Forest |
| 5 | 🎯 **Clustering** | K-Means (k=3) Wellness Personas with 3 visualisations |
| 6 | 🔍 **Association Rules** | Apriori mining with top-15 rules table + scatter plot |
| 7 | 📈 **Regression** | Linear Regression for Mood Score & Stress Reduction prediction |

---

## 🔧 Data Transformations (Applied Inside App)

All transformations are applied at runtime after loading the CSV — the source file is **not pre-processed**.

| Transformation | Formula |
|----------------|---------|
| `Challenge_Code` | `Yes → 1, No → 0` |
| `Subscription_Code` | `Free → 0, Premium → 1` *(ML only)* |
| `Stress_Reduction` | `Stress_Level_Before − Stress_Level_After` |
| `Engagement_Score` | `mean(Meditation, Yoga, Podcast, Audiobook minutes)` |

---

## 🤖 Machine Learning Models

### Classification (Page 4)
- **Target:** `Subscription_Type` (Free=0 / Premium=1)
- **Features:** Age, Meditation, Yoga, Streak Days, Mood Score, Stress After, App Usage, Satisfaction, Affirmations
- **Models:** Logistic Regression + Random Forest (80/20 split, `random_state=42`)
- **Outputs:** Accuracy, Precision, Recall, F1 — confusion matrix + feature importance

### Clustering (Page 5)
- **Features:** Meditation, Yoga, Affirmations, Stress After, Mood, Streak Days, App Usage
- **Method:** K-Means (k=3) with `StandardScaler`
- **Personas:** Casual Explorer | Stress Reliever | Mindfulness Champion

### Association Rules (Page 6)
- **Binary indicators:** High_Meditator, Watches_Reels, Listens_Podcasts, Reads_Audiobooks, Does_Yoga, Challenge_Participant, Premium_User, High_Affirmations
- **Parameters:** `min_support=0.3`, `min_confidence=0.6`
- **Library:** `mlxtend` *(graceful error message shown if not installed)*

### Regression (Page 7)
- **Model 1:** Predict `Mood_Score` from affirmations, meditation, sleep, stress, reels, streak
- **Model 2:** Predict `Stress_Reduction` from meditation, yoga, affirmations, sleep, app usage
- **Metrics:** R², MAE, RMSE + actual vs predicted scatter + coefficient table

---

## 🎛️ Sidebar Filters

The sidebar provides three multiselect filters — **Gender**, **Subscription Type**, and **Country/Region** — that dynamically filter data on the EDA and Correlation pages. All ML pages (Classification, Clustering, Association Rules, Regression) use the full unfiltered dataset.

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have **Python 3.9+** installed. Check your version:
```bash
python --version
```

### 2. Clone / Download the Project

```bash
git clone https://github.com/your-username/mindfulme-dashboard.git
cd mindfulme-dashboard
```

Or simply place `app.py`, `requirements.txt`, and `MindfulMe_Cleaned.csv` in the same folder.

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the Dataset

Place `MindfulMe_Cleaned.csv` in the **same directory** as `app.py`. The app loads it with:
```python
pd.read_csv("MindfulMe_Cleaned.csv")
```

### 6. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📋 Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| `User_ID` | Categorical | Unique user identifier |
| `Age` | Numerical | User age (18–65) |
| `Gender` | Categorical | Male / Female / Non-Binary |
| `Occupation` | Categorical | Student, Professional, Freelancer, etc. |
| `Country_Region` | Categorical | User's country |
| `Stress_Level_Before` | Numerical (1–10) | Stress before app session |
| `Stress_Level_After` | Numerical (1–10) | Stress after app session |
| `Mood_Score` | Numerical (1–10) | Daily mood rating |
| `Sleep_Hours` | Numerical | Average nightly sleep |
| `Meditation_Minutes_Per_Day` | Numerical | Daily meditation time |
| `Yoga_Minutes_Per_Day` | Numerical | Daily yoga time |
| `Affirmations_Per_Day` | Numerical | Daily affirmations consumed |
| `Reels_Viewed_Per_Day` | Numerical | Self-love reels watched |
| `Podcast_Listening_Minutes` | Numerical | Daily podcast time |
| `Audiobook_Listening_Minutes` | Numerical | Daily audiobook time |
| `App_Usage_Minutes_Per_Day` | Numerical | Total daily app time |
| `Streak_Days` | Numerical | Consecutive active days |
| `Challenge_Participation` | Categorical | Yes / No |
| `Subscription_Type` | Categorical | Free / Premium |
| `Productivity_Score` | Numerical (1–10) | Self-rated productivity |
| `Mindfulness_Score` | Numerical (1–100) | Composite mindfulness index |
| `User_Satisfaction` | Numerical (1–10) | Overall app satisfaction |

---

## 🎨 Design

- **Colour Theme:** Deep purple (`#7B4FBF`) and teal (`#1ABC9C`) on a dark background
- **Charts:** 100% Plotly — no Matplotlib or Seaborn
- **Insights:** Every chart is followed by a 2–3 sentence business insight using `st.caption()`
- **Responsive:** All layouts use Streamlit column grids with `use_container_width=True`

---

## ⚙️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: MindfulMe_Cleaned.csv` | Place the CSV in the same folder as `app.py` |
| `ModuleNotFoundError: mlxtend` | Run `pip install mlxtend` — the Association Rules page shows a warning if missing |
| `Trendline requires statsmodels` | Run `pip install statsmodels` for OLS trendlines in scatter plots |
| Charts not rendering | Ensure `plotly>=5.0.0` is installed |
| Slow first load on ML pages | First run trains models — subsequent loads use Streamlit's cache |

---

## 📚 Academic Context

This dashboard was built as part of the **IA-PBL (Industry Application — Project-Based Learning)** academic module. It demonstrates end-to-end data analytics including:

- Data wrangling and feature engineering
- Exploratory data analysis and visualisation
- Supervised learning (classification and regression)
- Unsupervised learning (clustering)
- Pattern mining (association rules)

---

## 📦 Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | Dashboard framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `plotly` | Interactive visualisations |
| `scikit-learn` | ML models + preprocessing |
| `mlxtend` | Apriori association rule mining |
| `statsmodels` | OLS trendlines in scatter plots |

---

*Built for IA-PBL Academic Project*
