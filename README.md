# 30-Day Hospital Readmission Risk Predictor

**Clinical AI Portfolio Project |**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://readmission-risk.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-KarkiAi-black)](https://github.com/KarkiAi/readmission-risk-predictor)

> Built with Synthea synthetic data, XGBoost, Logistic Regression, and SHAP explainability.
> Not for clinical use — demonstration only.

---

## What This Project Does

This machine learning pipeline predicts which patients are at risk of being 
readmitted to the hospital within 30 days of discharge. It uses synthetic 
patient data (Synthea), two ML models, and SHAP explainability to show 
not just the risk score — but why each patient received that score.

The live Streamlit dashboard allows a care coordinator to enter patient 
parameters at discharge and immediately see a risk tier (Low/Moderate/High) 
with a feature-level explanation.

---

## Model Performance (v2.0)

| Model | v1.0 AUC | v1.1 AUC | v2.0 AUC |
|---|---|---|---|
| Logistic Regression | 0.850 | 0.938 | **0.950** |
| XGBoost | 0.750 | 0.856 | **0.856** |

Trained on 101 Synthea synthetic inpatient encounters.
13 features including clinical, utilization, and SDOH variables.

---

## High-Risk Patient Profile

The model correctly identifies this patient as 99.5% readmission risk:

| Feature | Value |
|---|---|
| Age | 76, Male |
| Polypharmacy | Yes (5+ medications) |
| Active Conditions | 81 |
| High-Risk Diagnosis | Yes (CHF/Diabetes/COPD/CKD) |
| Prior Inpatient Stays | 15 |
| Prior Visits (6mo) | 6 |
| Low Income | Yes |
| High Procedure Burden | Yes |

---

## Features (V2 — 13 Total)

| Feature | Type | Source |
|---|---|---|
| AGE_AT_ENCOUNTER | Clinical | patients.csv |
| LENGTH_OF_STAY | Clinical | encounters.csv |
| GENDER_ENCODED | Demographic | patients.csv |
| POLYPHARMACY | Clinical | medications.csv |
| CONDITION_COUNT | Clinical | conditions.csv |
| HIGH_RISK_CONDITION | Clinical | conditions.csv |
| PRIOR_VISITS_6MO | Utilization | encounters.csv |
| PRIOR_INPATIENT_TOTAL | Utilization | encounters.csv |
| EMERGENCY_VISITS_6MO | Utilization | encounters.csv |
| LOW_INCOME | SDOH | patients.csv |
| HIGH_PROCEDURE_BURDEN | Clinical | procedures.csv |
| WINTER_DISCHARGE | Operational | encounters.csv |
| MARITAL_ENCODED | SDOH | patients.csv |

---

## Key Findings

- **PRIOR_INPATIENT_TOTAL** is the strongest predictor across all versions
- **Income gap confirmed:** readmitted patients average $45,246 vs 
  $133,870 for non-readmitted — strong SDOH signal
- **Emergency visits** 3x higher for readmitted patients (0.59 vs 0.18)
- **Logistic Regression outperforms XGBoost** on small dataset — 
  expected behavior; XGBoost advantage will emerge at 500+ encounters

---

## Clinical Implication

A care coordinator alert triggered at discharge for patients with high 
prior utilization + polypharmacy + low income + high-risk diagnosis would 
allow targeted post-discharge follow-up within 24-48 hours, potentially 
preventing costly readmissions. Social work referral should be considered 
for low-income patients given the strong income signal identified in V2.

---

## Tech Stack

Python • XGBoost • Scikit-learn • SHAP • Streamlit • Synthea • pandas • 
matplotlib • joblib • GitHub • Streamlit Community Cloud

---

## Project Structure

readmission-risk-predictor/

├── app/

│   └── app.py              # Streamlit dashboard

├── models/

│   ├── xgb_readmission_v2.pkl

│   └── lr_readmission_v2.pkl

├── data/                   # Synthea CSV files (not committed)

├── notebooks/              # Training notebooks

├── requirements.txt

└── README.md

---

## Model Updates & Changelog

### v2.0 — June 2026
- Added SDOH features: LOW_INCOME, EMERGENCY_VISITS_6MO,
  HIGH_PROCEDURE_BURDEN, WINTER_DISCHARGE, MARITAL_ENCODED
- Features increased from 8 to 13
- Added procedures.csv to pipeline
- LR AUC improved to 0.950
- Income gap identified: readmitted patients average $45,246 vs
  $133,870 for non-readmitted
- Emergency visits 3x higher for readmitted patients confirmed
- Fairness audit framework planned for v2.1

### v1.1 — June 2026
- Removed MED_COUNT after SHAP identified out-of-distribution failure
- Retained POLYPHARMACY binary flag — consistent with LACE score methodology
- AUC improved: LR 0.850 → 0.938, XGBoost 0.750 → 0.856

### v1.0 — June 2026
- Initial release with XGBoost and Logistic Regression
- 8 features, SHAP explainability integrated
- Deployed to Streamlit Community Cloud

---

## Known Limitations and Ethical Considerations

I built this project using Synthea synthetic patient data, which means 
the model has not been tested on real patient populations. This is an 
important limitation to acknowledge upfront.

As a medical graduate who has worked in hospitals and ICUs, I understand 
that healthcare data does not exist in a vacuum. It reflects decades of 
systemic inequities in how care is delivered, documented, and accessed 
across different communities.

In real-world clinical data, features like prior inpatient utilization — 
the strongest predictor in this model — are deeply influenced by 
socioeconomic factors, insurance status, and access to post-discharge 
support. A model that learns from this pattern without understanding the 
social cause can systematically mislabel marginalized patients.

Synthea generates race and ethnicity randomly rather than simulating 
real-world disparities, so this version cannot fully demonstrate 
fairness auditing on meaningful subgroups. Version 2.1 will address 
this with a formal subgroup fairness audit.

## Known Limitations

## Known Limitations

- Trained on 101 synthetic inpatient encounters (Synthea)
- Used 90-day readmission window during training to ensure 
  sufficient positive cases (30-day window yielded only 6 
  readmission events on current dataset)
- Active development — v2.1 will include additional QA, 
  feature refinement, and fairness audit framework
- Race/ethnicity fairness audit not meaningful on synthetic data
- Real-world validation required before any clinical use
- Not for clinical use — demonstration only
