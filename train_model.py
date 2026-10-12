"""Trains and compares ML models for loan approval prediction.

Run from the project root:   python train_model.py
Saves: models/model.pkl, models/metrics.json and plots in docs/screenshots/
"""
import json
import os

import joblib
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.utils import (METRICS_PATH, MODEL_PATH, PLOTS_DIR, TARGET,
                       build_preprocessor, load_and_clean)

df = load_and_clean()
X = df.drop(columns=[TARGET])
y = (df[TARGET] == "Y").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}

results, fitted = {}, {}
for name, clf in models.items():
    pipe = Pipeline([("prep", build_preprocessor(X)), ("model", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }
    fitted[name] = (pipe, pred)
    print(f"{name:20s}", results[name])

best = max(results, key=lambda k: results[k]["f1"])
best_pipe, best_pred = fitted[best]
print("\nBest model:", best)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
joblib.dump(best_pipe, MODEL_PATH, compress=3)
with open(METRICS_PATH, "w") as f:
    json.dump({"best_model": best, "results": results,
               "train_size": len(X_train), "test_size": len(X_test)}, f, indent=2)

# ---------------- Plots ----------------
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, best_pred),
                       display_labels=["Rejected", "Approved"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {best}")
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
names = list(results); x = np.arange(len(names)); w = 0.2
for i, m in enumerate(["accuracy", "precision", "recall", "f1"]):
    ax.bar(x + i * w, [results[n][m] for n in names], w, label=m.capitalize())
ax.set_xticks(x + 1.5 * w); ax.set_xticklabels(names, fontsize=8)
ax.set_ylim(0, 1.05); ax.set_title("Model Comparison"); ax.legend(fontsize=7, ncol=4, loc="lower center")
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150); plt.close()

rf = fitted["Random Forest"][0]
feat = rf.named_steps["prep"].get_feature_names_out()
imp = rf.named_steps["model"].feature_importances_
top = np.argsort(imp)[-8:]
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh([feat[i].split("__")[1] for i in top], imp[top], color="#2a6fb0")
ax.set_title("Top Feature Importances (Random Forest)")
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(4.5, 3.5))
df[TARGET].value_counts().plot(kind="bar", color=["#2a6fb0", "#c0504d"], ax=ax)
ax.set_title("Class Distribution (Y = Approved)"); ax.set_xlabel(""); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150); plt.close()
print("Saved model, metrics and plots.")
