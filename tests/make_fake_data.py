"""
FOR TESTING ONLY - generates fake trials in the same JSON shape as the
ClinicalTrials.gov API, so the pipeline can be tested offline.
Never use this data in the dashboard you show to anyone.
    python tests/make_fake_data.py
"""
import json, random
from pathlib import Path

random.seed(1)
conds = ["Alzheimer Disease", "Parkinson Disease", "Epilepsy", "Multiple Sclerosis",
         "Stroke", "Migraine", "Glioblastoma", "Amyotrophic Lateral Sclerosis"]
phases = [["PHASE1"], ["PHASE2"], ["PHASE3"], ["PHASE1", "PHASE2"], ["NA"], ["PHASE4"]]
countries = ["United States", "United Kingdom", "France", "Germany", "Italy", "China"]
reasons = ["Slow enrollment", "Lack of funding", "Sponsor decision", "Safety concerns", "Futility at interim"]

Path("data").mkdir(exist_ok=True)
with open("data/raw_trials.jsonl", "w") as f:
    for i in range(3000):
        ph = random.choice(phases)
        cls = random.choice(["INDUSTRY", "OTHER", "NIH"])
        n_sites = random.choice([1, 1, 2, 5, 10, 30])
        p_term = 0.08 + 0.10 * (ph == ["PHASE2"]) + 0.08 * (n_sites == 1) + 0.05 * (cls == "INDUSTRY")
        status = random.choice(["RECRUITING", "ACTIVE_NOT_RECRUITING"]) if random.random() < 0.15 else \
            ("TERMINATED" if random.random() < p_term else "COMPLETED")
        y = random.randint(2002, 2024)
        itype = random.choice(["DRUG", "DEVICE", "BEHAVIORAL", "BIOLOGICAL"])
        f.write(json.dumps({"protocolSection": {
            "identificationModule": {"nctId": f"NCT{i:08d}", "briefTitle": f"Fake trial {i}"},
            "statusModule": {"overallStatus": status,
                             "whyStopped": random.choice(reasons) if status == "TERMINATED" else None,
                             "startDateStruct": {"date": f"{y}-{random.randint(1,12):02d}"},
                             "primaryCompletionDateStruct": {"date": f"{y + random.randint(1,4)}-06"}},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": f"Sponsor {random.randint(1,40)}", "class": cls}},
            "conditionsModule": {"conditions": [random.choice(conds)]},
            "designModule": {"studyType": "INTERVENTIONAL", "phases": ph,
                             "enrollmentInfo": {"count": random.randint(10, 800), "type": "ACTUAL"},
                             "designInfo": {"allocation": random.choice(["RANDOMIZED", "NON_RANDOMIZED", "NA"]),
                                            "primaryPurpose": random.choice(["TREATMENT", "PREVENTION", "DIAGNOSTIC"]),
                                            "maskingInfo": {"masking": random.choice(["NONE", "DOUBLE", "QUADRUPLE"])}}},
            "armsInterventionsModule": {"armGroups": [{}] * random.randint(1, 4), "interventions": [{"type": itype}]},
            "contactsLocationsModule": {"locations": [{"country": random.choice(countries)} for _ in range(n_sites)]},
        }}) + "\n")
print("Wrote 3000 fake trials to data/raw_trials.jsonl")
