
---

# 📊 Customer Churn Predictor & Retention Dashboard

A machine learning web app that predicts which customers might cancel their subscriptions, explains why they are leaving, and finds the best discount strategy to maximize business profits.

---

## 🧾 Project Overview

Most basic machine learning models assume that any customer with a churn risk over 50% will leave. However, this ignores the real cost of business decisions: giving discounts to customers who were going to stay anyway wastes money, while missing high-value customers who leave loses revenue.

The **LoyaltyEngine** solves this problem by:

* **Cleaning Data Automatically:** Handles missing values and prepares customer data for training.
* **Smart Model Training:** Uses CatBoost to train on customer records while automatically handling class imbalance (since fewer people churn than stay).
* **Finding the Most Profitable Cutoff:** Tests 91 different decision thresholds to pick the single cutoff mark that saves the business the most money.
* **Explaining Every Prediction (SHAP):** Shows exactly which factors (like contract type or monthly fees) pushed a specific customer toward leaving or staying.

---

## 🌐 Live Demo

* **🎯 Streamlit Web App:** [Try the Live App Here](https://churn-diagnostics-decision-platform-liotf7qywhrjjxsvp6pumb.streamlit.app)

The live web dashboard lets you:

* Adjust discount costs and customer values with sliders to see the profit curve update in real time.
* Test individual customer profiles to see their churn risk score and top risk factors.
* Upload a full CSV file to get churn predictions and recommended retention offers for thousands of customers at once.

---

## 🧮 Dataset Details

* **Data Source:** [Telco Customer Churn Dataset (IBM / Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Target Column:** `Churn` (`Yes` / `No`)
* **Data Split:** 75% for training the model, 25% for testing performance.

**Features Included:**

* **Demographics:** Gender, Senior Citizen status, Partner, Dependents.
* **Services:** Phone service, internet type, online security, tech support, streaming services.
* **Billing & Account:** Contract length (Month-to-month, 1-Year, 2-Year), payment method, monthly charges, total charges, tenure (months with company).

---

## 🏗️ Repository Structure

* **`ChurnPredictor/`**
* **`.streamlit/`**
* `config.toml` — UI styling and dashboard settings


* **`data/`**
* `WA_Fn-UseC_-Telco-Customer-Churn.csv` — Customer dataset


* `app.py` — Streamlit web dashboard interface
* `engine.py` — Machine learning model, SHAP, and profit logic
* `requirements.txt` — List of Python packages needed
* `.gitignore` — Files Git should ignore
* `README.md` — Project documentation



**File Descriptions:**

* **`.streamlit/`** → Custom themes and appearance settings for the web app.
* **`data/`** → The raw telecom customer data.
* **`app.py`** → Frontend code for the interactive tabs, forms, charts, and file uploader.
* **`engine.py`** → Backend code that trains CatBoost, runs SHAP explanations, and calculates optimal profit.

---

## 📋 What Was Built

**1. Data Cleaning**

* Filled missing numerical values with medians and replaced empty text fields.
* Converted target churn labels into numbers (`1` for left, `0` for stayed).

**2. Model Training with CatBoost**

* Built the core pipeline using **CatBoost Classifier**, which works well with categorical data without heavy one-hot encoding.
* Balanced weights so the model pays equal attention to churners.

**3. Business Profit Optimization**

* Calculated the real cost of four outcomes:
* **Saved Customer (True Positive):** Kept customer revenue minus the discount cost.
* **Wasted Discount (False Positive):** Spent discount money on someone who was not leaving.
* **Safe Customer (True Negative):** Loyal customer left alone ($0 cost).
* **Missed Churner (False Negative):** Lost the entire customer value.


* Tested cutoffs from **5% to 95%** to find the threshold that generates maximum profit.

**4. Model Explanations (SHAP)**

* Used SHAP waterfall charts to show the exact dollar or feature reasons behind each customer's prediction.
* Built automated rules to suggest the right fix (e.g., offer a long-term plan discount or add tech support).

**5. Interactive Streamlit App**

* Built a clean 4-tab dashboard featuring profit simulations, account testing forms, demographic charts, and batch CSV predictions.

---
## 📊 Results & Key Findings

* **ROC-AUC Score:** **0.8482**, indicating strong model discrimination between retained and churning subscribers.
* **Churner Detection Rate (Recall):** Reached **95.93%** at the profit-optimized threshold (`0.21`), capturing almost all at-risk accounts.
* **Precision:** **39.61%** at the optimal threshold, balancing campaign reach against retention offer costs.
* **Business Value:** Shifting from the default `0.50` threshold to the cost-optimized `0.21` threshold maximized net cohort returns with **$2,089.00 in projected net savings**.

### Top Churn Reasons
* **Month-to-month contracts** were the biggest single driver of churn.
* **New subscribers (< 12 months tenure)** exhibited the highest risk of departure.
* **Digital Services:** Lack of Online Security and Tech Support strongly correlated with cancellations.
* **Payment Method:** Electronic Check users left at significantly higher rates than automated billing subscribers.
---

## 🛠️ Tools & Technologies

* **Language:** Python 3.10+
* **Machine Learning:** CatBoost, Scikit-Learn
* **Model Explainability:** SHAP
* **Data Handling:** Pandas, NumPy
* **App & Charts:** Streamlit, Matplotlib

---

## ⚙️ How to Run Locally

**1. Clone this repository**

```bash
git clone [https://github.com/MaheliN1218/Churn-Diagnostics-Decision-Platform.git](https://github.com/MaheliN1218/Churn-Diagnostics-Decision-Platform.git)
cd Churn-Diagnostics-Decision-Platform

```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv

```

* **On Windows:** `.venv\Scripts\activate`
* **On Mac/Linux:** `source .venv/bin/activate`

**3. Install packages**

```bash
pip install -r requirements.txt

```

**4. Start the app**

```bash
streamlit run app.py

```

---

## 📚 References

* **CatBoost:** Prokhorenkova et al. (2018). *CatBoost: unbiased boosting with categorical features*.
* **SHAP:** Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions*.
* **Cost-Sensitive Learning:** Verbeke et al. (2012). *Profit-driven data mining approach for churn prediction*.
* **Dataset:** IBM Telco Customer Churn dataset on Kaggle.