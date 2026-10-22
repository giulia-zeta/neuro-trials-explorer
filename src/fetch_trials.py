"""
Step 1 - Download neurology clinical trials from ClinicalTrials.gov.

Uses the public ClinicalTrials.gov API (v2): https://clinicaltrials.gov/data-api/api
No API key is needed.

We download interventional trials for a set of neurological conditions and
save the raw JSON records to data/raw_trials.jsonl (one trial per line).

Run from the project folder:
    python src/fetch_trials.py
    python src/fetch_trials.py --max-studies 5000     # smaller, faster test
"""

import argparse
import json
import time
from pathlib import Path

import requests

API_URL = "https://clinicaltrials.gov/api/v2/studies"

# Neurological conditions to include. ClinicalTrials.gov search syntax: OR between terms.
CONDITIONS = [
    "Alzheimer Disease", "Dementia", "Parkinson Disease", "Epilepsy",
    "Multiple Sclerosis", "Stroke", "Amyotrophic Lateral Sclerosis",
    "Migraine", "Huntington Disease", "Neuropathic Pain", "Traumatic Brain Injury",
    "Spinal Cord Injury", "Glioblastoma",
]

# Only the parts of each record we need (keeps the download small).
FIELDS = [
    "protocolSection.identificationModule",
    "protocolSection.statusModule",
    "protocolSection.sponsorCollaboratorsModule",
    "protocolSection.conditionsModule",
    "protocolSection.designModule",
    "protocolSection.armsInterventionsModule",
    "protocolSection.contactsLocationsModule",
]


def fetch(max_studies: int, out_path: Path) -> int:
    params = {
        "query.cond": " OR ".join(f'"{c}"' for c in CONDITIONS),
        "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
        "fields": ",".join(FIELDS),
        "pageSize": 1000,          # maximum allowed by the API
        "format": "json",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        while n < max_studies:
            r = requests.get(API_URL, params=params, timeout=60)
            r.raise_for_status()
            payload = r.json()
            for study in payload.get("studies", []):
                f.write(json.dumps(study) + "\n")
                n += 1
                if n >= max_studies:
                    break
            print(f"Downloaded {n} trials...")
            token = payload.get("nextPageToken")
            if not token:
                break
            params["pageToken"] = token
            time.sleep(0.5)        # be polite to the public API
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-studies", type=int, default=30000)
    parser.add_argument("--out", default="data/raw_trials.jsonl")
    args = parser.parse_args()
    total = fetch(args.max_studies, Path(args.out))
    print(f"Done: saved {total} trials to {args.out}")
