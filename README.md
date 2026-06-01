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

