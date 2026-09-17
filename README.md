 # 🏦 Loan Approval Predictor

## 📌 Project Overview

The **Loan Approval Predictor** is a Machine Learning project that predicts whether a loan application is likely to be **approved or rejected** based on different details provided by the applicant.

The project uses historical loan application data to train a Machine Learning classification model. After training, users can enter applicant information through a simple interface and receive a prediction.

> **Note:** This project is for educational purposes. Its predictions should not be used as the sole basis for real-world lending decisions.

---

## 🎯 Objectives

The main objectives of this project are:

* To understand the basics of Machine Learning.
* To perform data preprocessing.
* To handle missing values in a dataset.
* To convert categorical data into numerical data.
* To train a classification model.
* To evaluate the performance of the model.
* To create a simple user-friendly interface.
* To predict loan approval based on applicant information.

---

## 🧠 Machine Learning Concept

This project is a **Binary Classification** problem.

The model predicts one of two possible outcomes:

```text
Approved
Not Approved
```

### Algorithm

The project can use:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

The performance of the models can be compared using evaluation metrics such as accuracy.

---

## 📊 Dataset

The dataset contains information about previous loan applications.

Some of the important features are:

| Feature           | Description                            |
| ----------------- | -------------------------------------- |
| Gender            | Gender of the applicant                |
| Married           | Marital status                         |
| Dependents        | Number of dependents                   |
| Education         | Education level                        |
| Self_Employed     | Whether the applicant is self-employed |
| ApplicantIncome   | Applicant's income                     |
| CoapplicantIncome | Co-applicant's income                  |
| LoanAmount        | Requested loan amount                  |
| Loan_Amount_Term  | Loan repayment term                    |
| Credit_History    | Credit history information             |
| Property_Area     | Area of the property                   |
| Loan_Status       | Loan approval result                   |

`Loan_Status` is the target variable.

---

## 🔄 Project Workflow

```text
              Dataset
                 │
                 ↓
        Data Preprocessing
                 │
        ┌────────┴────────┐
        │                 │
   Missing Values     Data Encoding
        │                 │
        └────────┬────────┘
                 ↓
          Train/Test Split
                 │
                 ↓
        Machine Learning Model
                 │
                 ↓
             Prediction
                 │
        ┌────────┴────────┐
        ↓                 ↓
     Approved        Not Approved
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation
* **NumPy** – Numerical operations
* **Scikit-learn** – Machine Learning
* **Matplotlib** – Data visualization
* **Streamlit** – Web interface
* **Joblib** – Saving the trained model
* **Jupyter Notebook / VS Code** – Development environment

---

## 📁 Project Structure

```text
Loan_Approval_Predictor/
│
├── dataset/
│   └── loan_data.csv
│
├── model/
│   └── loan_model.pkl
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project folder

```bash
cd Loan_Approval_Predictor
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the libraries manually:

```bash
pip install pandas numpy matplotlib scikit-learn streamlit joblib
```

---

## ▶️ How to Run the Project

### Step 1: Train the model

Run:

```bash
python train_model.py
```

This will preprocess the dataset, train the Machine Learning model, evaluate it, and save the trained model.

---

### Step 2: Run the Streamlit application

Run:

```bash
streamlit run app.py
```

The application will open in your web browser.

---

## 🖥️ Application

The application allows the user to enter information such as:

```text
Gender
Marital Status
Education
Dependents
Applicant Income
Co-applicant Income
Loan Amount
Loan Term
Credit History
Property Area
```

After entering the information, the user can click:

```text
PREDICT
```

The system then displays the predicted result:

```text
✅ Loan Approved
```

or

```text
❌ Loan Not Approved
```

---

## 📈 Model Evaluation

The model can be evaluated using:

### Accuracy

Accuracy measures the proportion of correct predictions.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Confusion Matrix

A confusion matrix helps visualize:

* Correct approvals
* Correct rejections
* Incorrect approvals
* Incorrect rejections

Other metrics that can be considered include:

* Precision
* Recall
* F1-Score

The actual performance values should be reported after training the model on the selected dataset.

---

## ✨ Features

* Simple and easy-to-use interface
* Machine Learning based prediction
* Handles applicant information
* Data preprocessing
* Classification model
* Model evaluation
* Interactive Streamlit application
* Easy to run locally

---

## 🚀 Future Scope

The project can be improved in the future by:

* Using larger and more diverse datasets.
* Comparing additional Machine Learning algorithms.
* Improving feature engineering.
* Adding probability/confidence information with appropriate interpretation.
* Adding better data visualizations.
* Deploying the application online.
* Adding secure database storage.
* Adding model monitoring and fairness checks for real-world use.

---

## ⚠️ Limitations

* The prediction depends on the quality and representativeness of the training dataset.
* A Machine Learning model can make incorrect predictions.
* Historical data may contain biases that can affect model results.
* The model should not be treated as a guaranteed loan decision.
* Real-world loan approval involves additional information, policies, verification, and regulatory requirements.

---

## 🎓 Educational Purpose

This project was developed as a **Fundamentals of Artificial Intelligence and Machine Learning** project to demonstrate:

* Data preprocessing
* Classification
* Model training
* Model evaluation
* Prediction
* Basic AI/ML application development

---

## All programs in google collab file
https://colab.research.google.com/drive/1k0StjK_6Rp6WEN44zk6S2QQ17HlD6RDp#scrollTo=y9qL4tPmf9Y6

---

## 👨‍💻 Author

**Mohit Thakur**
**25MIM10126**
Student Project
Fundamentals of Artificial Intelligence and Machine Learning

---

## 📜 License

This project is created for educational purposes.
