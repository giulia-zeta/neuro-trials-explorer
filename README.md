# Neurology Clinical Trials Explorer

An interactive dashboard and prediction model built on public data from
[ClinicalTrials.gov](https://clinicaltrials.gov). It looks at interventional trials
in neurology (Alzheimer's, Parkinson's, epilepsy, MS, stroke, ALS and more) and asks:

- How are neurology trials designed, and how long do they take?
- Which trials are terminated before completion, and why?
- Using only information available **when a trial is designed**, can we estimate
  its risk of early termination?

Built with Python (pandas, scikit-learn), Plotly and Streamlit.

## How it works

| Step | Script | What it does |
|---|---|---|
| 1 | `src/fetch_trials.py` | Downloads trial records from the ClinicalTrials.gov API (v2) |
| 2 | `src/clean_trials.py` | Flattens the nested JSON into one table, groups conditions, derives duration and outcome |
| 3 | `src/train_model.py` | Trains and cross-validates models predicting early termination |
| 4 | `app.py` | Streamlit dashboard: filters, charts, reasons for stopping, risk estimator |

## Running it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/fetch_trials.py         # a few minutes; use --max-studies 5000 for a quick test
python src/clean_trials.py
python src/train_model.py
streamlit run app.py
```

## Modelling choices

- **Outcome:** terminated vs completed, among trials that have finished.
- **Features:** phase, condition, sponsor type, randomisation, blinding, purpose,
  number of arms, sites and countries, intervention type, UK site, start year.
- **No data leakage:** actual enrolment, end dates and stated reasons for stopping
  are only known after a trial runs, so they are excluded from the model.
- **Validation:** 5-fold stratified cross-validation, reporting ROC AUC and average
  precision (more informative than accuracy, because most trials complete).
- **Two models:** logistic regression (interpretable coefficients) and gradient
  boosting (captures non-linear effects); the better one is used in the dashboard.

## Limitations

- Registry data is self-reported and sometimes incomplete or out of date.
- "Reasons for stopping" are free text, grouped here by simple keyword matching.
- The model shows associations in historical data; it is not a decision tool.

## Ideas for next steps

- Predict enrolment shortfall (actual vs planned enrolment) as a second outcome
- Add site-level views (countries, number of sites over time)
- Use text from trial titles and descriptions as extra features

---
Author: Giulia Zuccarini, PhD
