# Customer Churn Prediction (Data Analytics Project)

## Project Overview

Customer churn is a major problem for subscription-based businesses such as telecom, banking, and SaaS companies. Predicting which customers are likely to leave helps organizations take preventive actions and improve customer retention.

This project uses **Machine Learning classification models** to predict whether a customer will churn based on several behavioral and service-related features.

The workflow includes:

* Data generation and preparation
* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training
* Model evaluation
* Visualization of results

---

# Objective

The goal of this project is to build a machine learning system that can:

* Predict which customers are likely to **stop using a service**
* Identify important factors that contribute to **customer churn**
* Help businesses take actions to **improve customer retention**

---

# Dataset

This project uses a **synthetic telecom customer dataset containing 5000 records**.

The dataset simulates real-world telecom customer data with features related to service usage and billing behavior.

### Features

| Feature          | Description                                   |
| ---------------- | --------------------------------------------- |
| Tenure           | Number of months customer stayed with company |
| MonthlyCharges   | Monthly payment amount                        |
| TotalCharges     | Total amount paid                             |
| NumProducts      | Number of services used                       |
| Contract         | Contract type                                 |
| InternetService  | Type of internet service                      |
| TechSupport      | Whether technical support is included         |
| OnlineBackup     | Whether online backup is enabled              |
| SeniorCitizen    | Whether customer is a senior citizen          |
| Partner          | Whether customer has a partner                |
| PaperlessBilling | Paperless billing enabled                     |
| PaymentMethod    | Payment method used                           |
| SupportCalls     | Number of support calls                       |
| Churn            | Target variable (0 = Stayed, 1 = Churned)     |

---

# Project Workflow

## 1. Data Generation

A synthetic telecom dataset with **5000 customers** is generated using Python.

## 2. Exploratory Data Analysis (EDA)

EDA is performed to understand customer behavior and identify patterns related to churn.

Visualizations include:

* Churn distribution
* Tenure distribution
* Monthly charges distribution
* Contract type vs churn rate
* Internet service vs churn rate
* Support calls vs churn
* Correlation heatmap

---

## 3. Feature Engineering

Additional features were created to improve model performance.

Examples:

* Charges per month
* High support call flag
* Long tenure flag

Categorical features were encoded using **Label Encoding**.

---

## 4. Machine Learning Models

Three classification models were trained and evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting

These models were selected because they are commonly used for **customer churn prediction problems**.

---

# Model Evaluation

The models were evaluated using the following metrics:

| Metric    | Description                            |
| --------- | -------------------------------------- |
| Accuracy  | Overall prediction correctness         |
| Recall    | Ability to detect churn customers      |
| Precision | Correctness of predicted churn         |
| F1 Score  | Balance between precision and recall   |
| ROC-AUC   | Model's ability to distinguish classes |

Recall is particularly important because **missing churn customers can result in revenue loss**.

---

# Project Structure

```
Customer_Churn_Prediction
│
├── churn_prediction.py
├── telecom_churn.csv
├── model_results.csv
├── customer-churn-prediction.ipynb
│
├── outputs
│   ├── fig1_eda.png
│   ├── fig2_evaluation.png
│   ├── fig3_feature_importance.png
│   └── fig4_business_insights.png
│
└── README.md
```

---

# Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# How to Run the Project

Run the Python script:

```bash
python churn_prediction.py
```

This will generate:

* Dataset file
* Model results
* EDA visualizations
* Model evaluation plots
* Feature importance analysis

---

# Key Insights

Some important patterns discovered during analysis:

* Customers with **short tenure** are more likely to churn.
* Customers with **many support calls** have higher churn probability.
* **Month-to-month contracts** show higher churn rates.
* Customers with **technical support services** tend to stay longer.

---

# Future Improvements

Possible improvements for this project include:

* Using real telecom datasets
* Implementing **XGBoost or LightGBM models**
* Hyperparameter tuning
* Deploying the model using **Flask or FastAPI**
* Building an interactive **dashboard for churn prediction**

---

# Author

**Mitesh Gorad**

Master of Computer Science (MCS)
Interested in Data Analytics, Embedded Systems, and Machine Learning.

---
