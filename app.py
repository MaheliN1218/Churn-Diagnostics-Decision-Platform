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



# page 2

elif st.session_state.active_tab == "Account Diagnostics":
    st.markdown("### Account Risk Diagnostic")

    c_form, c_results = st.columns([1.1, 1.1])

    input_data = {}
    with c_form:
        st.markdown("""
        <div style="background:#111827; border:1px solid #1F2937; border-top:2px solid #0D9488; border-radius:8px; padding:16px; margin-bottom:14px;">
            <div style="font-size:11px; font-weight:700; color:#2DD4BF; font-family:'Space Grotesk',monospace; text-transform:uppercase; margin-bottom:8px;">
                Plan & Financials
            </div>
        """, unsafe_allow_html=True)

        input_data['tenure'] = st.slider("Account Tenure (Months)", 1, 72, 4)
        input_data['MonthlyCharges'] = st.slider("Monthly Plan Charge ($)", 18.0, 120.0, 89.5)
        input_data['Contract'] = st.selectbox("Contract Agreement", ["Month-to-month", "One year", "Two year"])
        input_data['PaymentMethod'] = st.selectbox("Payment Channel", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        input_data['TotalCharges'] = float(input_data['tenure'] * input_data['MonthlyCharges'])

        st.markdown("</div>", unsafe_allow_html=True)

        # Service Subscriptions (Tick Boxes)
        st.markdown("""
        <div style="background:#111827; border:1px solid #1F2937; border-top:2px solid #06B6D4; border-radius:8px; padding:16px; margin-bottom:14px;">
            <div style="font-size:11px; font-weight:700; color:#06B6D4; font-family:'Space Grotesk',monospace; text-transform:uppercase; margin-bottom:8px;">
                Network & Digital Services
            </div>
        """, unsafe_allow_html=True)

        input_data['InternetService'] = st.selectbox("Internet Infrastructure", ["Fiber optic", "DSL", "No"])

        has_internet = (input_data['InternetService'] != "No")

        sb1, sb2 = st.columns(2)
        with sb1:
            online_sec = st.checkbox("Online Security Suite", value=False, disabled=not has_internet)
            online_bkp = st.checkbox("Cloud Backup Service", value=True, disabled=not has_internet)
            dev_prot = st.checkbox("Device Protection Plan", value=False, disabled=not has_internet)
        with sb2:
            tech_sup = st.checkbox("Dedicated Tech Support", value=False, disabled=not has_internet)
            strm_tv = st.checkbox("Streaming TV Bundle", value=False, disabled=not has_internet)
            strm_mov = st.checkbox("Streaming Movies Bundle", value=False, disabled=not has_internet)

        # Convert bools to dataset categorical format
        input_data['OnlineSecurity'] = "Yes" if online_sec and has_internet else (
            "No internet service" if not has_internet else "No")
        input_data['OnlineBackup'] = "Yes" if online_bkp and has_internet else (
            "No internet service" if not has_internet else "No")
        input_data['DeviceProtection'] = "Yes" if dev_prot and has_internet else (
            "No internet service" if not has_internet else "No")
        input_data['TechSupport'] = "Yes" if tech_sup and has_internet else (
            "No internet service" if not has_internet else "No")
        input_data['StreamingTV'] = "Yes" if strm_tv and has_internet else (
            "No internet service" if not has_internet else "No")
        input_data['StreamingMovies'] = "Yes" if strm_mov and has_internet else (
            "No internet service" if not has_internet else "No")

        st.markdown("</div>", unsafe_allow_html=True)

        # Demographics & Billing Settings
        st.markdown("""
        <div style="background:#111827; border:1px solid #1F2937; border-top:2px solid #F97316; border-radius:8px; padding:16px; margin-bottom:14px;">
            <div style="font-size:11px; font-weight:700; color:#F97316; font-family:'Space Grotesk',monospace; text-transform:uppercase; margin-bottom:8px;">
                Subscriber Profile & Billing Mode
            </div>
        """, unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            input_data['gender'] = st.selectbox("Gender", ["Female", "Male"])
            is_senior = st.checkbox("Senior Citizen Account", value=False)
            is_paperless = st.checkbox("Paperless Invoicing", value=True)
        with col_g2:
            has_partner = st.checkbox("Has Partner", value=False)
            has_dependents = st.checkbox("Has Dependents", value=False)
            has_multilines = st.checkbox("Multiple Phone Lines", value=False)

        input_data['SeniorCitizen'] = 1 if is_senior else 0
        input_data['PaperlessBilling'] = "Yes" if is_paperless else "No"
        input_data['Partner'] = "Yes" if has_partner else "No"
        input_data['Dependents'] = "Yes" if has_dependents else "No"
        input_data['MultipleLines'] = "Yes" if has_multilines else "No"

        st.markdown("</div>", unsafe_allow_html=True)

        sample_df = pd.DataFrame([input_data])[engine.X.columns]
        for col in engine.cat_cols:
            sample_df[col] = sample_df[col].astype(str)
        for col in engine.num_cols:
            sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce')

        prob = float(engine.model.predict_proba(sample_df)[0][1])
        is_churn = prob >= engine.optimal_threshold

    with c_results:
        theme_color = "#F43F5E" if is_churn else "#0D9488"
        badge_bg = "rgba(244, 63, 94, 0.12)" if is_churn else "rgba(13, 148, 136, 0.12)"
        status_text = "HIGH RISK: ACTION REQUIRED" if is_churn else "STABLE ACCOUNT"

        st.markdown(
            f"""
            <div style="background-color:#111827; border:1px solid {theme_color}; border-radius:8px; padding:18px; text-align:center; margin-bottom:14px;">
                <span style="color:#64748B; font-size:11px; font-family:'Space Grotesk',monospace; font-weight:700;">PREDICTED CHURN PROBABILITY</span>
                <h1 style="color:{theme_color}; font-family:'Space Grotesk',monospace; font-size:38px; font-weight:700; margin:6px 0;">{prob * 100:.1f}%</h1>
                <span style="background-color:{badge_bg}; color:{theme_color}; padding:4px 10px; border-radius:4px; font-weight:600; font-size:11px; font-family:'Space Grotesk',monospace;">{status_text}</span>
                <p style="margin-top:8px; font-size:11.5px; color:#64748B;">Cutoff Threshold: <strong style="color:#FFFFFF;">{engine.optimal_threshold:.2f}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Prescribed Retention Action")
        if not is_churn:
            st.success("Stable Account: Standard operational engagement. No discount budget required.")
        else:
            if prob >= 0.70 and input_data['MonthlyCharges'] >= 80:
                st.info(r"Tier 1 - VIP Package: 20% billing reduction for 6 months + renewal outreach (\$35 Budget).")
            elif input_data['Contract'] == "Month-to-month":
                st.info(
                    r"Tier 2 - Annual Incentive: \$15 monthly credit for migrating to an annual agreement (\$15 Budget).")
            elif input_data['TechSupport'] == "No" and input_data['InternetService'] != "No":
                st.info(r"Tier 3 - Support Bundle: 3 months complimentary VIP Tech Support (\$8 Budget).")
            else:
                st.info(r"Tier 4 - Statement Credit: \$10 one-time billing adjustment (\$10 Budget).")

        st.markdown("---")
        st.markdown("#### Root-Cause Attribution (SHAP)")
        shap_val = engine.explain_single(sample_df)
        fig = plt.figure(figsize=(6.5, 3.2))
        shap.plots.waterfall(shap_val, max_display=7, show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


