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
    xgb = joblib.load('models/xgb_readmission_v21.pkl')
    lr  = joblib.load('models/lr_readmission_v21.pkl')
    return xgb, lr

xgb, lr = load_model()

FEATURES = [
    'AGE_AT_ENCOUNTER',
    'LENGTH_OF_STAY',
    'GENDER_ENCODED',
    'POLYPHARMACY',
    'CONDITION_COUNT',
    'HIGH_RISK_CONDITION',
    'PRIOR_VISITS_6MO',
    'PRIOR_INPATIENT_TOTAL',
    'EMERGENCY_VISITS_6MO',
    'LOW_INCOME',
    'MARITAL_ENCODED',
    'SOCIAL_ISOLATION'
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
polypharmacy_input = st.sidebar.checkbox("Taking 5 or more medications", value=True)
emergency_visits = st.sidebar.slider("Emergency Visits (last 6 months)", 0, 10, 0)
low_income = st.sidebar.checkbox("Low Income Household (under $50,000)", value=False)
married = st.sidebar.checkbox("Married / Has Partner", value=False)
social_isolation = st.sidebar.checkbox("Social Isolation (documented)", value=False)
ed_admission = st.sidebar.checkbox("Admitted via Emergency Department (Acuity)", value=False)
condition_count = st.sidebar.slider("Active Condition Count", 0, 150, 20)
prior_visits = st.sidebar.slider("Prior Visits (last 6 months)", 0, 20, 3)
prior_inpatient = st.sidebar.slider("Prior Inpatient Stays (total)", 0, 30, 1)
high_risk = st.sidebar.checkbox("High-Risk Diagnosis (CHF, COPD, Diabetes, CKD)")

# ── Derive features ──────────────────────────────────────────
gender_encoded = 1 if gender == "Male" else 0
polypharmacy = 1 if polypharmacy_input else 0
emergency_visits_6mo = emergency_visits
low_income_flag = 1 if low_income else 0
marital_encoded = 1 if married else 0
social_isolation_flag = 1 if social_isolation else 0
high_risk_flag = 1 if high_risk else 0

input_data = pd.DataFrame([{
    'AGE_AT_ENCOUNTER': age,
    'LENGTH_OF_STAY': los,
    'GENDER_ENCODED': gender_encoded,
    'POLYPHARMACY': polypharmacy,
    'CONDITION_COUNT': condition_count,
    'HIGH_RISK_CONDITION': high_risk_flag,
    'PRIOR_VISITS_6MO': prior_visits,
    'PRIOR_INPATIENT_TOTAL': prior_inpatient,
    'EMERGENCY_VISITS_6MO': emergency_visits_6mo,
    'LOW_INCOME': low_income_flag,
    'MARITAL_ENCODED': marital_encoded,
    'SOCIAL_ISOLATION': social_isolation_flag
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

# ── LACE Score ───────────────────────────────────────────────
def lace_L(los_days):
    if los_days < 1: return 0
    elif los_days == 1: return 1
    elif los_days == 2: return 2
    elif los_days == 3: return 3
    elif 4 <= los_days <= 6: return 4
    elif 7 <= los_days <= 13: return 5
    else: return 7

lace_A = 3 if ed_admission else 0

def lace_C(cond_count):
    if cond_count == 0: return 0
    elif cond_count <= 5: return 1
    elif cond_count <= 10: return 2
    elif cond_count <= 15: return 3
    else: return 5

def lace_E(ed_visits):
    return min(ed_visits, 4)

lace_total = lace_L(los) + lace_A + lace_C(condition_count) + lace_E(emergency_visits)

st.subheader("📐 LACE Index (Clinical Baseline)")
lcol1, lcol2 = st.columns(2)
with lcol1:
    st.metric("LACE Score", f"{lace_total} / 19")
with lcol2:
    if lace_total >= 10:
        st.markdown("### 🔴 HIGH RISK (LACE ≥ 10)")
    elif lace_total >= 5:
        st.markdown("### 🟡 MODERATE RISK")
    else:
        st.markdown("### 🟢 LOW RISK")

st.caption("""
LACE: Length of stay (L) + Acuity of admission (A) + Comorbidities (C) + 
ED visits in prior 6 months (E). Note: C is approximated from active 
condition count; the validated LACE index uses the Charlson Comorbidity 
Index. Displayed as a clinical baseline for comparison with ML predictions.
""")
st.divider()

st.subheader("🔍 Why This Risk Score? (SHAP Explanation)")

explainer = shap.Explainer(xgb)
shap_values = explainer(input_data)

# Extract SHAP values
vals = shap_values[0].values
names = shap_values[0].feature_names

# Table below chart
shap_df = pd.DataFrame({
    'Feature': names,
    'Impact': vals
}).sort_values('Impact', ascending=False)

shap_df['Direction'] = shap_df['Impact'].apply(
    lambda x: '🔴 Increases Risk' if x > 0 else '🔵 Decreases Risk'
)
shap_df['Impact'] = shap_df['Impact'].round(3)

st.dataframe(shap_df, use_container_width=True)



# ── Clinical interpretation ──────────────────────────────────
st.divider()
st.subheader("📋 Clinical Summary")

st.markdown(f"""
| Parameter | Value |
|---|---|
| Age | {age} years |
| Length of Stay | {los} days |
| Gender | {gender} |
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