"""
Step 4 - Interactive dashboard.

Run from the project folder:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Neuro Trials Explorer", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("data/trials_clean.csv", parse_dates=["start_date", "end_date"])
    return df[df["start_year"].between(2000, 2026)]


df = load_data()

st.title("Neurology Clinical Trials Explorer")
st.caption("Interventional neurology trials registered on ClinicalTrials.gov: "
           "how they are designed, how they recruit, and which ones stop early.")

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
groups = st.sidebar.multiselect("Condition", sorted(df["condition_group"].unique()))
phases = st.sidebar.multiselect("Phase", sorted(df["phase"].unique()))
sponsors = st.sidebar.multiselect("Sponsor type", sorted(df["sponsor_class"].dropna().unique()))
years = st.sidebar.slider("Start year", 2000, 2026, (2010, 2026))
uk_only = st.sidebar.checkbox("Only trials with a UK site")

f = df[df["start_year"].between(*years)]
if groups:
    f = f[f["condition_group"].isin(groups)]
if phases:
    f = f[f["phase"].isin(phases)]
if sponsors:
    f = f[f["sponsor_class"].isin(sponsors)]
if uk_only:
    f = f[f["has_uk_site"]]

closed = f[f["status"].isin(["COMPLETED", "TERMINATED"])]

# ---------------- Headline numbers ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trials", f"{len(f):,}")
c2.metric("Median enrolment", f"{f['enrollment'].median():.0f}" if len(f) else "–")
c3.metric("Terminated early",
          f"{closed['terminated'].mean():.1%}" if len(closed) else "–",
          help="Share of finished trials (completed or terminated) that were terminated.")
c4.metric("Median duration", f"{f['duration_months'].median():.0f} months" if len(f) else "–")

tab1, tab2, tab3 = st.tabs(["Overview", "Why trials stop", "Termination risk model"])

# ---------------- Tab 1: overview ----------------
with tab1:
    left, right = st.columns(2)
    per_year = f.groupby(["start_year", "phase"]).size().reset_index(name="trials")
    left.plotly_chart(px.bar(per_year, x="start_year", y="trials", color="phase",
                             title="Trials started per year, by phase"), width="stretch")
    by_group = f["condition_group"].value_counts().reset_index()
    by_group.columns = ["condition", "trials"]
    right.plotly_chart(px.bar(by_group, x="trials", y="condition", orientation="h",
                              title="Trials by condition"), width="stretch")

    left, right = st.columns(2)
    left.plotly_chart(px.box(f.dropna(subset=["duration_months"]), x="phase", y="duration_months",
                             points=False, title="Trial duration by phase (months)"),
                      width="stretch")
    top_sponsors = f["sponsor"].value_counts().head(15).reset_index()
    top_sponsors.columns = ["sponsor", "trials"]
    right.plotly_chart(px.bar(top_sponsors, x="trials", y="sponsor", orientation="h",
                              title="Most active sponsors"), width="stretch")

# ---------------- Tab 2: why trials stop ----------------
with tab2:
    if len(closed):
        rate = (closed.groupby("phase")["terminated"].mean().reset_index()
                .rename(columns={"terminated": "termination_rate"}))
        st.plotly_chart(px.bar(rate, x="phase", y="termination_rate",
                               title="Share of finished trials terminated early, by phase")
                        .update_yaxes(tickformat=".0%"), width="stretch")

    stopped = f[f["status"] == "TERMINATED"]
    reasons = {
        "Recruitment / enrolment": "enrol|recruit|accrual|participants",
        "Funding": "fund|budget|financ",
        "Sponsor / business decision": "sponsor|business|strateg|company",
        "Safety": "safety|adverse|toxic",
        "Efficacy / futility": "efficacy|futility|interim",
    }
    text = stopped["why_stopped"].fillna("").str.lower()
    counts = pd.DataFrame({"reason": list(reasons),
                           "trials": [text.str.contains(p).sum() for p in reasons.values()]})
    st.plotly_chart(px.bar(counts, x="trials", y="reason", orientation="h",
                           title=f"Stated reasons among {len(stopped):,} terminated trials "
                                 "(keyword match, a trial can match several)"),
                    width="stretch")
    st.dataframe(stopped[["nct_id", "title", "phase", "sponsor_class", "why_stopped"]]
                 .head(200), width="stretch", hide_index=True)

# ---------------- Tab 3: model ----------------
with tab3:
    if not Path("models/metrics.json").exists():
        st.info("Run `python src/train_model.py` first to train the model.")
    else:
        m = json.loads(Path("models/metrics.json").read_text())
        st.markdown(
            f"Trained on **{m['n_trials']:,}** finished trials "
            f"({m['termination_rate']:.1%} terminated), using only information available "
            "**when the trial is designed** (no actual enrolment or end dates, to avoid data leakage). "
            "Scores are from 5-fold cross-validation.")
        st.table(pd.DataFrame(m["models"]).T.rename(
            columns={"roc_auc": "ROC AUC", "avg_precision": "Average precision"}))
        top = pd.DataFrame(m["top_logistic_features"])
        st.plotly_chart(px.bar(top, x="coefficient", y="feature", orientation="h",
                               title="Strongest predictors (logistic regression, "
                                     "positive = higher termination risk)"),
                        width="stretch")

        st.subheader("Estimate the termination risk of a new trial design")
        model = joblib.load("models/termination_model.joblib")
        a, b, c = st.columns(3)
        row = {
            "phase": a.selectbox("Phase", sorted(df["phase"].unique())),
            "condition_group": a.selectbox("Condition", sorted(df["condition_group"].unique())),
            "sponsor_class": a.selectbox("Sponsor type", sorted(df["sponsor_class"].dropna().unique())),
            "allocation": b.selectbox("Allocation", sorted(df["allocation"].dropna().unique())),
            "masking": b.selectbox("Masking", sorted(df["masking"].dropna().unique())),
            "primary_purpose": b.selectbox("Purpose", sorted(df["primary_purpose"].dropna().unique())),
            "n_arms": c.number_input("Number of arms", 1, 10, 2),
            "n_sites": c.number_input("Number of sites", 1, 500, 10),
            "n_countries": c.number_input("Number of countries", 1, 50, 1),
            "start_year": 2026,
            "is_drug_trial": int(c.checkbox("Drug / biological", True)),
            "is_device_trial": int(c.checkbox("Device", False)),
            "has_uk_site": int(c.checkbox("UK site", True)),
        }
        risk = model.predict_proba(pd.DataFrame([row]))[0, 1]
        st.metric("Estimated risk of early termination", f"{risk:.0%}")
        st.caption("A teaching model, not a clinical tool: it shows which design features "
                   "are associated with early termination in historical data.")

st.divider()
st.caption("Data: ClinicalTrials.gov API v2. Built by Giulia Zuccarini.")
