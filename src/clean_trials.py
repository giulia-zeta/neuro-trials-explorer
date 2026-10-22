"""
Step 2 - Turn the raw nested JSON into one clean table (one row per trial).

Reads  data/raw_trials.jsonl
Writes data/trials_clean.csv

Run from the project folder:
    python src/clean_trials.py
"""

import json
from pathlib import Path

import pandas as pd

# Map free-text condition names to broader groups (first match wins).
CONDITION_GROUPS = {
    "Alzheimer's & dementia": ["alzheimer", "dementia", "cognitive impairment"],
    "Parkinson's": ["parkinson"],
    "Epilepsy": ["epilep", "seizure"],
    "Multiple sclerosis": ["multiple sclerosis"],
    "Stroke": ["stroke", "cerebrovascular"],
    "ALS / motor neurone disease": ["amyotrophic", "motor neuron"],
    "Migraine & headache": ["migraine", "headache"],
    "Huntington's": ["huntington"],
    "Pain": ["neuropathic pain", "neuralgia", "neuropathy"],
    "Brain & spinal injury": ["brain injur", "spinal cord injur", "concussion"],
    "Brain tumours": ["glioblastoma", "glioma", "brain tumo"],
}


def get(d, *keys, default=None):
    """Safely walk a nested dictionary: get(d, 'a', 'b') -> d['a']['b'] or default."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def condition_group(conditions):
    text = " ".join(conditions).lower()
    for group, keywords in CONDITION_GROUPS.items():
        if any(k in text for k in keywords):
            return group
    return "Other neurological"


def phase_label(phases):
    """Combine the phases list into one readable label, e.g. ['PHASE1','PHASE2'] -> 'Phase 1/2'."""
    if not phases:
        return "Not applicable"
    if phases == ["NA"]:
        return "Not applicable"
    nums = sorted(p.replace("EARLY_PHASE1", "PHASE0").replace("PHASE", "") for p in phases)
    nums = ["Early 1" if n == "0" else n for n in nums]
    return "Phase " + "/".join(nums)


def parse_date(s):
    # Dates come as 'YYYY-MM' or 'YYYY-MM-DD'
    return pd.to_datetime(s, errors="coerce") if s else pd.NaT


def flatten(study: dict) -> dict:
    p = study.get("protocolSection", {})
    conditions = get(p, "conditionsModule", "conditions", default=[]) or []
    phases = get(p, "designModule", "phases", default=[]) or []
    locations = get(p, "contactsLocationsModule", "locations", default=[]) or []
    countries = {loc.get("country") for loc in locations if loc.get("country")}
    interventions = get(p, "armsInterventionsModule", "interventions", default=[]) or []
    intervention_types = sorted({i.get("type", "") for i in interventions if i.get("type")})

    start = parse_date(get(p, "statusModule", "startDateStruct", "date"))
    end = parse_date(get(p, "statusModule", "primaryCompletionDateStruct", "date"))
    if pd.isna(end):
        end = parse_date(get(p, "statusModule", "completionDateStruct", "date"))

    return {
        "nct_id": get(p, "identificationModule", "nctId"),
        "title": get(p, "identificationModule", "briefTitle"),
        "conditions": "; ".join(conditions),
        "condition_group": condition_group(conditions),
        "status": get(p, "statusModule", "overallStatus"),
        "why_stopped": get(p, "statusModule", "whyStopped"),
        "phase": phase_label(phases),
        "start_date": start,
        "end_date": end,
        "enrollment": get(p, "designModule", "enrollmentInfo", "count"),
        "enrollment_type": get(p, "designModule", "enrollmentInfo", "type"),
        "sponsor": get(p, "sponsorCollaboratorsModule", "leadSponsor", "name"),
        "sponsor_class": get(p, "sponsorCollaboratorsModule", "leadSponsor", "class"),
        "allocation": get(p, "designModule", "designInfo", "allocation"),
        "masking": get(p, "designModule", "designInfo", "maskingInfo", "masking"),
        "primary_purpose": get(p, "designModule", "designInfo", "primaryPurpose"),
        "n_arms": len(get(p, "armsInterventionsModule", "armGroups", default=[]) or []),
        "intervention_types": "; ".join(intervention_types),
        "is_drug_trial": "DRUG" in intervention_types or "BIOLOGICAL" in intervention_types,
        "is_device_trial": "DEVICE" in intervention_types,
        "n_sites": len(locations),
        "n_countries": len(countries),
        "has_uk_site": "United Kingdom" in countries,
    }


def main(raw_path="data/raw_trials.jsonl", out_path="data/trials_clean.csv"):
    with open(raw_path, encoding="utf-8") as f:
        rows = [flatten(json.loads(line)) for line in f if line.strip()]
    df = pd.DataFrame(rows).drop_duplicates("nct_id")

    # Derived columns
    df["start_year"] = df["start_date"].dt.year
    df["duration_months"] = (df["end_date"] - df["start_date"]).dt.days / 30.44
    df.loc[df["duration_months"] < 0, "duration_months"] = None
    df["terminated"] = (df["status"] == "TERMINATED").astype(int)
    df["stopped_for_recruitment"] = df["why_stopped"].fillna("").str.lower().str.contains(
        "enrol|recruit|accrual|participants"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} trials to {out_path}")
    print(df["status"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
