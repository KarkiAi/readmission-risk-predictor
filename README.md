# readmission-risk-predictor


=== CLINICAL INTERPRETATION ===

Model Performance:
- Logistic Regression AUC: 0.850 (best performer on small dataset)
- XGBoost AUC: 0.750 (expected to improve significantly with larger dataset)

Key Finding:
The model identifies PRIOR_INPATIENT_TOTAL as the strongest predictor
of 30-day readmission — patients with more historical inpatient stays
are dramatically more likely to be readmitted. This aligns with clinical
literature showing that frequent utilizers drive disproportionate 
readmission rates.

High-Risk Patient Profile (99.4% predicted risk):
- Age 76, male
- 695 cumulative medications, 84 active conditions
- 17 prior inpatient stays
- High-risk diagnosis present (CHF/Diabetes/COPD/CKD)

Clinical Implication:
A care coordinator alert triggered at discharge for patients matching
this profile — high prior utilization + polypharmacy + high-risk diagnosis
— would allow targeted post-discharge follow-up within 24-48 hours,
potentially preventing costly readmissions.

Limitation:
Model trained on 101 synthetic inpatient encounters (Synthea).
Performance metrics expected to improve substantially with real-world
or larger synthetic datasets. XGBoost advantage over logistic regression
will emerge at 1000+ encounters.

## Model Updates & Changelog

### v1.1 — June 2026
- Removed MED_COUNT feature after QA identified out-of-distribution 
  failure on cumulative medication counts
- Retained POLYPHARMACY binary flag (5+ meds) — clinically equivalent 
  to LACE score methodology and more stable on small datasets
- Finding: SHAP waterfall showed MED_COUNT incorrectly suppressing 
  risk scores for high-risk patients
- Fix: Binary polypharmacy flag eliminates continuous range instability
- AUC improved: Logistic Regression 0.850 → 0.938, XGBoost 0.750 → 0.856

### v1.0 — June 2026
- Initial release with XGBoost (AUC 0.75) and Logistic Regression (AUC 0.85)
- Trained on 101 Synthea synthetic inpatient encounters
- SHAP explainability integrated
- Deployed to Streamlit Community Cloud

## Known Limitations
- Trained on 101 synthetic inpatient encounters (Synthea)
- Predicts 90-day readmission window
- Performance expected to improve with larger dataset (500+ encounters)
- Not for clinical use — demonstration only
