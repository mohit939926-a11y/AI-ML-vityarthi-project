"""Shared helpers used by train_model.py, app.py and the tests."""
import json
import os
import subprocess
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "loan_data.csv")
MODEL_PATH = os.path.join(ROOT, "models", "model.pkl")
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")
PLOTS_DIR = os.path.join(ROOT, "docs", "screenshots")

FEATURES = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "Property_Area"]
TARGET = "Loan_Status"


def load_and_clean(path=DATA_PATH):
    """Reads the CSV and fills missing values (mode for text, median for numbers)."""
    df = pd.read_csv(path)
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
    for col in df.select_dtypes(exclude="number"):
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include="number"):
        df[col] = df[col].fillna(df[col].median())
    return df


def build_preprocessor(X):
    """One-hot encodes text columns and scales numeric columns."""
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


def load_model():
    """Loads models/model.pkl. If the file is missing, trains it first (takes a few seconds)."""
    if not os.path.exists(MODEL_PATH):
        subprocess.run([sys.executable, os.path.join(ROOT, "train_model.py")], cwd=ROOT, check=True)
    return joblib.load(MODEL_PATH)


def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def predict_loan(model, applicant: dict):
    """Returns (prediction 1/0, approval probability) for one applicant dictionary."""
    row = pd.DataFrame([applicant])[FEATURES]
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])
    return pred, prob
