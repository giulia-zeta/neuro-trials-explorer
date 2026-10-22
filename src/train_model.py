"""
Step 3 - Predict which trials are terminated early.

Question: using only information known when a trial is designed and registered,
can we estimate its risk of being terminated before completion?

Important: we deliberately leave out anything only known AFTER the trial runs
(actual enrolment, end date, reason for stopping). Using those would be
"data leakage" - the model would look great but be useless in practice.

Reads  data/trials_clean.csv
Writes models/termination_model.joblib and models/metrics.json

Run from the project folder:
    python src/train_model.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["phase", "condition_group", "sponsor_class", "allocation",
               "masking", "primary_purpose"]
NUMERIC = ["n_arms", "n_sites", "n_countries", "start_year"]
BOOLEAN = ["is_drug_trial", "is_device_trial", "has_uk_site"]
FEATURES = CATEGORICAL + NUMERIC + BOOLEAN


def load(path="data/trials_clean.csv"):
    df = pd.read_csv(path)
    # Only trials with a final outcome: completed vs terminated
    df = df[df["status"].isin(["COMPLETED", "TERMINATED"])].copy()
    df = df[df["start_year"] >= 2000]
    for c in CATEGORICAL:
        df[c] = df[c].fillna("Unknown").astype(str)
    for c in NUMERIC:
        df[c] = df[c].fillna(df[c].median())
    for c in BOOLEAN:
        df[c] = df[c].astype(str).str.lower().eq("true").astype(int)
    return df


def make_models():
    pre_linear = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC + BOOLEAN),
    ])
    pre_tree = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
        ("num", "passthrough", NUMERIC + BOOLEAN),
    ], sparse_threshold=0)
    return {
        "Logistic regression": Pipeline([
            ("prep", pre_linear),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "Gradient boosting": Pipeline([
            ("prep", pre_tree),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                   random_state=0)),
        ]),
    }


def main():
    df = load()
    X, y = df[FEATURES], df["terminated"]
    print(f"{len(df)} closed trials; {y.mean():.1%} terminated")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    results, best_name, best_auc = {}, None, -1
    for name, model in make_models().items():
        # Out-of-fold predictions: each trial is scored by a model that never saw it
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, proba)
        ap = average_precision_score(y, proba)
        results[name] = {"roc_auc": round(auc, 3), "avg_precision": round(ap, 3)}
        print(f"{name}: ROC AUC = {auc:.3f}, average precision = {ap:.3f}")
        if auc > best_auc:
            best_name, best_auc = name, auc

    # Refit the best model on all data and save it for the dashboard
    best = make_models()[best_name].fit(X, y)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best, "models/termination_model.joblib")

    # Interpretable view: logistic regression coefficients (odds ratios)
    lr = make_models()["Logistic regression"].fit(X, y)
    def readable(name):
        name = name.split("__", 1)[-1]          # drop the 'cat__' / 'num__' prefix
        for c in CATEGORICAL:                   # 'phase_Phase 2' -> 'phase: Phase 2'
            if name.startswith(c + "_"):
                return f"{c.replace('_', ' ')}: {name[len(c) + 1:]}"
        return name.replace("_", " ")

    names = [readable(n) for n in lr.named_steps["prep"].get_feature_names_out()]
    coefs = lr.named_steps["clf"].coef_[0]
    top = (pd.DataFrame({"feature": names, "coefficient": coefs})
           .sort_values("coefficient", key=abs, ascending=False).head(15))

    metrics = {
        "n_trials": int(len(df)),
        "termination_rate": round(float(y.mean()), 3),
        "models": results,
        "best_model": best_name,
        "top_logistic_features": top.round(3).to_dict(orient="records"),
    }
    Path("models/metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved best model ({best_name}) to models/termination_model.joblib")


if __name__ == "__main__":
    main()
