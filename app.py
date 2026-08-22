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


# 3. sidebar

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Strategy"


def set_active_tab(selected_tab_name):
    st.session_state.active_tab = selected_tab_name


with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 12px; margin-bottom: 16px; border-bottom: 1px solid #1E293B;">
        <span style="font-size: 10px; font-weight: 700; color: #94A3B8; font-family: 'Space Grotesk', monospace; text-transform: uppercase;">Control Center</span>
        <h2 style="font-size: 16px; font-weight: 700; color: #FFFFFF; margin: 4px 0 0 0;">Menu</h2>
    </div>
    """, unsafe_allow_html=True)

    tab_options = [
        "Strategy",
        "Account Diagnostics",
        "Demographics",
        "Batch Evaluation"
    ]

    for i, tab in enumerate(tab_options):
        is_active = (st.session_state.active_tab == tab)
        btn_type = "primary" if is_active else "secondary"
        st.button(
            label=tab,
            key=f"sidebar_btn_{i}",
            type=btn_type,
            use_container_width=True,
            on_click=set_active_tab,
            args=(tab,)
        )


# 4. top hero banner

st.markdown("""
<div class="hero-box">
    <span class="hero-badge">Cost-Matrix Optimization</span>
    <h1 style="color: #FFFFFF; margin: 0 0 4px 0;">Churn Diagnostics & Decision Platform</h1>
    <p style="color: #94A3B8; font-size: 13px; margin: 0; line-height: 1.5;">
        Threshold tuning, financial impact modeling, and root-cause risk diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)


# page 1: strategy and profit curve

if st.session_state.active_tab == "Strategy":

    if "cust_val" not in st.session_state:
        st.session_state.cust_val = 70.0
    if "inc_cost" not in st.session_state:
        st.session_state.inc_cost = 15.0
    if "acc_rate" not in st.session_state:
        st.session_state.acc_rate = 0.65

    optimal_t = engine.optimize_threshold(
        incentive_cost=st.session_state.inc_cost,
        customer_value=st.session_state.cust_val,
        acceptance_rate=st.session_state.acc_rate
    )

    st.markdown("### Profit Optimization Curve")
    c_left, c_right = st.columns([1.35, 1])

    with c_left:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(engine.thresholds, engine.profits, color="#0D9488", lw=2.0, label="Expected Net Profit ($)")
        ax.axhline(0, color="#334155", linestyle=":", lw=1.0, label="Break-Even ($0)")
        ax.axvline(optimal_t, color="#F97316", linestyle="--", lw=1.5, label=f"Optimal Cutoff ({optimal_t:.2f})")
        ax.fill_between(engine.thresholds, engine.profits, 0, color="#0D9488", alpha=0.10)

        peak_profit = max(engine.profits)
        ax.set_ylim(bottom=-6000, top=peak_profit * 1.35)
        ax.yaxis.set_major_formatter("${x:,.0f}")
        ax.set_xlabel("Decision Threshold", fontsize=9, fontweight="bold")
        ax.set_ylabel("Cohort Net Value ($)", fontsize=9, fontweight="bold")
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c_right:
        net_val = (st.session_state.cust_val * st.session_state.acc_rate) - st.session_state.inc_cost
        st.markdown(f"""
        <div style="background:#111827; border:1px solid #1F2937; border-radius:8px; padding:18px; font-size:12.5px; line-height:1.7;">
            <div style="font-weight:700; color:#FFFFFF; font-size:12px; margin-bottom:8px; font-family:'Space Grotesk',monospace; text-transform:uppercase;">
                Expected Value Matrix
            </div>
            • <strong style="color:#2DD4BF;">True Positive (Saved Churner):</strong> (${st.session_state.cust_val:.0f} × {st.session_state.acc_rate * 100:.0f}%) - ${st.session_state.inc_cost:.0f} = <strong>+${net_val:.2f}</strong><br>
            • <strong style="color:#F43F5E;">False Positive (Wasted Discount):</strong> Sent to loyal = <strong>-${st.session_state.inc_cost:.2f}</strong><br>
            • <strong style="color:#64748B;">False Negative (Missed Churner):</strong> Lost account equity = <strong>-${st.session_state.cust_val:.2f}</strong><br><br>
            <em>The decision threshold shifts dynamically to maximize total net profit.</em>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    st.markdown("### Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'''<div class="metric-card accent">
                <div class="card-title">Optimal Cutoff</div>
                <div class="card-metric" style="color:#F97316;">{optimal_t:.2f}</div>
            </div>''', unsafe_allow_html=True)
    with k2:
        st.markdown(
            f'''<div class="metric-card">
                <div class="card-title">Churn Recall</div>
                <div class="card-metric" style="color:#2DD4BF;">{engine.metrics["recall"] * 100:.1f}%</div>
            </div>''', unsafe_allow_html=True)
    with k3:
        st.markdown(
            f'''<div class="metric-card">
                <div class="card-title">Model ROC-AUC</div>
                <div class="card-metric">{engine.metrics["roc_auc"]:.3f}</div>
            </div>''', unsafe_allow_html=True)
    with k4:
        st.markdown(
            f'''<div class="metric-card">
                <div class="card-title">Projected Net Savings</div>
                <div class="card-metric">${engine.metrics["projected_savings"]:,.0f}</div>
            </div>''', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    st.markdown("### Simulation Parameters")
    with st.container():
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.session_state.cust_val = st.slider(
                "Customer Lifetime Value (LTV) ($)",
                min_value=20.0, max_value=200.0, value=float(st.session_state.cust_val), step=5.0
            )
        with f_col2:
            st.session_state.inc_cost = st.slider(
                "Retention Offer Unit Cost ($)",
                min_value=5.0, max_value=50.0, value=float(st.session_state.inc_cost), step=1.0
            )
        with f_col3:
            st.session_state.acc_rate = st.slider(
                "Offer Acceptance Rate (%)",
                min_value=10, max_value=90, value=int(st.session_state.acc_rate * 100), step=5
            ) / 100.0


