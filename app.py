"""Loan Approval Predictor - Streamlit web app.   Run:  streamlit run app.py"""
import streamlit as st

from src.utils import load_metrics, load_model, predict_loan

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def get_model():
    return load_model()


model = get_model()
metrics = load_metrics()

st.title("🏦 Loan Approval Predictor")
st.caption("Fundamentals in AIML | Mohit Thakur (25MIM10126)")
st.write("Enter the applicant details below to predict whether the loan is likely to be approved.")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
with col2:
    applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Coapplicant Income (monthly)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=120, step=10)
    loan_term = st.selectbox("Loan Amount Term (months)", [360, 180, 480, 300, 240, 120])
    credit_history = st.selectbox(
        "Credit History", [1, 0],
        format_func=lambda v: "1 - Good credit history" if v == 1 else "0 - Poor / no credit history")

if st.button("Predict Loan Status", type="primary"):
    pred, prob = predict_loan(model, {
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history, "Property_Area": property_area})
    if pred == 1:
        st.success(f"✅ Loan is likely to be APPROVED (probability: {prob:.1%})")
    else:
        st.error(f"❌ Loan is likely to be REJECTED (approval probability: {prob:.1%})")
    st.progress(prob)

with st.expander("About the model"):
    best = metrics["best_model"]; r = metrics["results"][best]
    st.write(f"**Model used:** {best}")
    st.write(f"Accuracy: {r['accuracy']:.1%} | Precision: {r['precision']:.1%} | "
             f"Recall: {r['recall']:.1%} | F1-score: {r['f1']:.1%}")
    st.caption("Educational project. Predictions are not real financial advice.")
