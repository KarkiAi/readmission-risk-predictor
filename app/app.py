import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="30-Day Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    xgb = joblib.load('models/xgb_readmission.pkl')
    lr  = joblib.load('models/lr_readmission.pkl')
    return xgb, lr

xgb, lr = load_model()

FEATURES = [
    'AGE_AT_ENCOUNTER', 'LENGTH_OF_STAY', 'GENDER_ENCODED',
    'MED_COUNT', 'POLYPHARMACY', 'CONDITION_COUNT',
    'HIGH_RISK_CONDITION', 'PRIOR_VISITS_6MO', 'PRIOR_INPATIENT_TOTAL'
]

# ── Header ───────────────────────────────────────────────────
st.title("🏥 30-Day Readmission Risk Predictor")
st.markdown("""
> **Clinical AI Portfolio Project** | Built with Synthea synthetic data, 
> XGBoost, and SHAP explainability  
> *Not for clinical use — demonstration only*
""")

st.divider()

# ── Sidebar inputs ───────────────────────────────────────────
st.sidebar.header("Patient Parameters")
st.sidebar.markdown("Adjust values to match patient profile at discharge")

age = st.sidebar.slider("Age at Encounter", 18, 95, 65)
los = st.sidebar.slider("Length of Stay (days)", 0, 30, 3)
gender = st.sidebar.radio("Gender", ["Female", "Male"])
med_count = st.sidebar.slider("Cumulative Medication Count", 0, 800, 50)
condition_count = st.sidebar.slider("Active Condition Count", 0, 150, 20)
prior_visits = st.sidebar.slider("Prior Visits (last 6 months)", 0, 20, 3)
prior_inpatient = st.sidebar.slider("Prior Inpatient Stays (total)", 0, 30, 1)
high_risk = st.sidebar.checkbox("High-Risk Diagnosis (CHF, COPD, Diabetes, CKD)")

# ── Derive features ──────────────────────────────────────────
gender_encoded = 1 if gender == "Male" else 0
polypharmacy = 1 if med_count >= 5 else 0
high_risk_flag = 1 if high_risk else 0

input_data = pd.DataFrame([{
    'AGE_AT_ENCOUNTER': age,
    'LENGTH_OF_STAY': los,
    'GENDER_ENCODED': gender_encoded,
    'MED_COUNT': med_count,
    'POLYPHARMACY': polypharmacy,
    'CONDITION_COUNT': condition_count,
    'HIGH_RISK_CONDITION': high_risk_flag,
    'PRIOR_VISITS_6MO': prior_visits,
    'PRIOR_INPATIENT_TOTAL': prior_inpatient
}])

# ── Predictions ──────────────────────────────────────────────
xgb_prob = xgb.predict_proba(input_data)[0][1]
lr_prob  = lr.predict_proba(input_data)[0][1]

# Risk tier
if xgb_prob >= 0.7:
    risk_label = "🔴 HIGH RISK"
    risk_color = "red"
elif xgb_prob >= 0.4:
    risk_label = "🟡 MODERATE RISK"
    risk_color = "orange"
else:
    risk_label = "🟢 LOW RISK"
    risk_color = "green"

# ── Results layout ───────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("XGBoost Risk Score", f"{xgb_prob:.1%}")
with col2:
    st.metric("Logistic Regression Score", f"{lr_prob:.1%}")
with col3:
    st.markdown(f"### {risk_label}")

st.divider()

# ── SHAP explanation ─────────────────────────────────────────
st.subheader("🔍 Why This Risk Score? (SHAP Explanation)")

explainer = shap.Explainer(xgb)
shap_values = explainer(input_data)

fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.waterfall(shap_values[0], max_display=9, show=False)
st.pyplot(fig)
plt.close()

# ── Clinical interpretation ──────────────────────────────────
st.divider()
st.subheader("📋 Clinical Summary")

st.markdown(f"""
| Parameter | Value |
|---|---|
| Age | {age} years |
| Length of Stay | {los} days |
| Gender | {gender} |
| Medication Count | {med_count} |
| Polypharmacy | {'Yes' if polypharmacy else 'No'} |
| Active Conditions | {condition_count} |
| High-Risk Diagnosis | {'Yes' if high_risk else 'No'} |
| Prior Visits (6mo) | {prior_visits} |
| Prior Inpatient Stays | {prior_inpatient} |
""")

if xgb_prob >= 0.7:
    st.warning("""
    ⚠️ **Recommended Action:** Flag for care coordinator follow-up within 24 hours 
    of discharge. Consider transition care management (TCM) referral.
    """)
elif xgb_prob >= 0.4:
    st.info("""
    ℹ️ **Recommended Action:** Schedule follow-up call within 72 hours. 
    Review discharge instructions and medication reconciliation.
    """)
else:
    st.success("""
    ✅ **Recommended Action:** Standard discharge protocol. 
    Routine follow-up appointment within 7-14 days.
    """)

st.divider()
st.caption("Built by Shhai Karki | Synthea synthetic data | Not for clinical use")