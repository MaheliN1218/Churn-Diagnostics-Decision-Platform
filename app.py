import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from engine import LoyaltyEngine


# 1. page setup

st.set_page_config(
    page_title="Churn Diagnostics & Decision Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)





# 2. data pipeline

@st.cache_resource
def load_telco_engine():
    path_a = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    path_b = os.path.join("data", "Telco-Customer-Churn.csv")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

    if os.path.exists(path_a):
        df = pd.read_csv(path_a)
    elif os.path.exists(path_b):
        df = pd.read_csv(path_b)
    else:
        df = pd.read_csv(url)

    engine = LoyaltyEngine(df, target_col="Churn", id_cols=["customerID", "PhoneService"])
    engine.train(incentive_cost=15.0, customer_value=70.0, acceptance_rate=0.65)
    return engine, df


with st.spinner("Initializing Model Engine..."):
    engine, raw_df = load_telco_engine()

raw_df['Churn_Binary'] = raw_df['Churn'].apply(
    lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true', 'churned'] else 0
)
