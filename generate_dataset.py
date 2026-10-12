"""Generates a synthetic loan application dataset (data/loan_data.csv).

The columns follow the well-known 'Loan Prediction' dataset layout so the
project can be re-run on a real Kaggle dataset by simply replacing loan_data.csv.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1000

gender = rng.choice(["Male", "Female"], N, p=[0.8, 0.2])
married = rng.choice(["Yes", "No"], N, p=[0.65, 0.35])
dependents = rng.choice(["0", "1", "2", "3+"], N, p=[0.57, 0.17, 0.17, 0.09])
education = rng.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
self_employed = rng.choice(["Yes", "No"], N, p=[0.14, 0.86])
property_area = rng.choice(["Urban", "Semiurban", "Rural"], N, p=[0.33, 0.38, 0.29])

income = rng.lognormal(mean=8.5, sigma=0.5, size=N).round(0)
income = np.where(education == "Graduate", income * 1.25, income).round(0)
coapp = np.where(married == "Yes", rng.lognormal(7.4, 0.9, N), rng.lognormal(5.0, 1.0, N) * 0.3).round(0)
loan_amount = (rng.normal(0.035, 0.012, N) * (income + coapp) * 12 / 10).clip(9, 600).round(0)
term = rng.choice([360, 180, 480, 300, 240, 120], N, p=[0.80, 0.07, 0.05, 0.04, 0.02, 0.02])
credit = rng.choice([1, 0], N, p=[0.80, 0.20])

# Approval logic: credit history dominates, affordability and a bit of noise matter too
emi_ratio = (loan_amount * 1000 / term) / (income + coapp + 1)
score = (
    3.2 * credit
    - 4.0 * emi_ratio
    + 0.5 * (education == "Graduate")
    + 0.4 * (married == "Yes")
    + 0.5 * (property_area == "Semiurban")
    + 0.0003 * (income + coapp)
    - 1.6
    + rng.normal(0, 0.9, N)
)
threshold = np.quantile(score, 0.31)  # ~69% approvals, similar to real loan datasets
status = np.where(score > threshold, "Y", "N")

df = pd.DataFrame({
    "Loan_ID": [f"LP{1000+i}" for i in range(N)],
    "Gender": gender, "Married": married, "Dependents": dependents,
    "Education": education, "Self_Employed": self_employed,
    "ApplicantIncome": income.astype(int), "CoapplicantIncome": coapp.astype(int),
    "LoanAmount": loan_amount.astype(int), "Loan_Amount_Term": term,
    "Credit_History": credit, "Property_Area": property_area, "Loan_Status": status,
})
import os
os.makedirs("data", exist_ok=True)
df.to_csv(os.path.join("data", "loan_data.csv"), index=False)
print(df.shape)
print(df["Loan_Status"].value_counts())
