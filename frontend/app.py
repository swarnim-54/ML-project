"""
frontend/app.py
---------------
Streamlit UI for Loan Default Prediction.
Calls FastAPI backend via HTTP — no direct model usage.
"""

import os
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700; color: #1F4E79;
        border-bottom: 3px solid #1F4E79; padding-bottom: 0.4rem;
        margin-bottom: 1.5rem;
    }
    .card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1.2rem; border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .result-default {
        background: #fff0f0; border: 2px solid #c0392b;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .result-safe {
        background: #f0fff4; border: 2px solid #27ae60;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .result-medium {
        background: #fffbe6; border: 2px solid #f39c12;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .prob-bar-wrap {
        background: #e9ecef; border-radius: 6px;
        height: 14px; margin-top: 6px;
    }
    .section-head {
        font-size: 1.05rem; font-weight: 600;
        color: #1F4E79; margin: 1rem 0 0.5rem 0;
        border-left: 4px solid #1F4E79; padding-left: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Preset values from actual dataset ─────────────────────────────────────────
# High-risk borrower (from test set defaulter)
HIGH_RISK = {
    "term": 36, "int_rate": 7.51, "emp_length": 5.0,
    "annual_inc": 125000.0, "dti": 1.08, "delinq_2yrs": 0.0,
    "inq_last_6mths": 0.0, "open_acc": 6.0, "pub_rec": 0.0,
    "revol_bal": 740.0, "revol_util": 3.2, "total_acc": 10.0,
    "total_rec_prncp": 3767.98, "total_rec_int": 890.17,
    "last_pymnt_amnt": 933.33, "loan_amnt": 25000.0,
    "installment": 930.0, "total_pymnt": 4658.0,
    "earliest_cr_line_M": 1, "earliest_cr_line_Y": 2001,
    "issue_d_M": 6, "issue_d_Y": 2013,
    "last_pymnt_d_M": 3, "last_pymnt_d_Y": 2014,
    "last_credit_pull_d_M": 4, "last_credit_pull_d_Y": 2014,
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "purpose": "debt_consolidation"
}

# Low-risk borrower (from test set non-defaulter)
LOW_RISK = {
    "term": 36, "int_rate": 7.37, "emp_length": 7.0,
    "annual_inc": 39996.0, "dti": 6.39, "delinq_2yrs": 0.0,
    "inq_last_6mths": 0.0, "open_acc": 7.0, "pub_rec": 0.0,
    "revol_bal": 409.0, "revol_util": 0.02, "total_acc": 12.0,
    "total_rec_prncp": 3000.0, "total_rec_int": 353.03,
    "last_pymnt_amnt": 93.78, "loan_amnt": 3000.0,
    "installment": 93.0, "total_pymnt": 3353.0,
    "earliest_cr_line_M": 8, "earliest_cr_line_Y": 2003,
    "issue_d_M": 9, "issue_d_Y": 2012,
    "last_pymnt_d_M": 11, "last_pymnt_d_Y": 2015,
    "last_credit_pull_d_M": 1, "last_credit_pull_d_Y": 2016,
    "home_ownership": "MORTGAGE",
    "verification_status": "Not Verified",
    "purpose": "credit_card"
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Loan Default Predictor")
    st.markdown("---")

    # API health
    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        status = h.get("status", "unknown")
        threshold = h.get("threshold", "N/A")
        dot = "🟢" if status == "healthy" else "🔴"
        st.markdown(f"{dot} **API:** {status.upper()}")
        st.caption(f"Decision threshold: {threshold}")
    except Exception:
        st.markdown("🔴 **API: OFFLINE**")
        st.warning(f"Start the backend:\n```\nuvicorn api.main:app --port 8000\n```")

    st.markdown("---")
    st.markdown("**Quick Fill**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔴 High Risk", use_container_width=True):
            st.session_state["preset"] = "high"
            st.rerun()
    with col_b:
        if st.button("🟢 Low Risk", use_container_width=True):
            st.session_state["preset"] = "low"
            st.rerun()

    st.markdown("---")
    st.markdown("""
**About**

XGBoost model trained on 37,000+ real loan records.
Threshold optimised using business cost:
- Missed default → $5,000 loss
- False rejection → $200 loss

Results include SHAP-based explanations.
""")


# ── Apply preset ──────────────────────────────────────────────────────────────
preset = st.session_state.get("preset", "low")
D = HIGH_RISK if preset == "high" else LOW_RISK

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏦 Loan Default Risk Assessment</div>',
            unsafe_allow_html=True)
st.caption("Fill in the loan application details below and click Assess Risk.")

with st.form("loan_form"):

    # ── Section 1: Loan details ───────────────────────────────────────────────
    st.markdown('<div class="section-head">Loan Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        loan_amnt   = st.number_input("Loan Amount ($)", 500.0, 40000.0,
                                       float(D["loan_amnt"]), 100.0)
        term        = st.selectbox("Term (months)", [36, 60],
                                    index=0 if D["term"]==36 else 1)
    with c2:
        int_rate    = st.number_input("Interest Rate (%)", 5.0, 30.0,
                                       float(D["int_rate"]), 0.1)
        installment = st.number_input("Monthly Installment ($)", 10.0, 2000.0,
                                       float(D["installment"]), 1.0)
    with c3:
        purpose = st.selectbox("Loan Purpose", [
            "debt_consolidation","credit_card","home_improvement",
            "major_purchase","small_business","other","medical",
            "car","moving","vacation","house","educational",
            "wedding","renewable_energy"
        ], index=["debt_consolidation","credit_card","home_improvement",
                   "major_purchase","small_business","other","medical",
                   "car","moving","vacation","house","educational",
                   "wedding","renewable_energy"].index(D["purpose"]))
        total_pymnt = st.number_input("Total Payment Received ($)", 0.0, 50000.0,
                                       float(D["total_pymnt"]), 10.0)

    # ── Section 2: Borrower profile ───────────────────────────────────────────
    st.markdown('<div class="section-head">Borrower Profile</div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        annual_inc = st.number_input("Annual Income ($)", 5000.0, 500000.0,
                                      float(D["annual_inc"]), 1000.0)
        emp_length = st.number_input("Employment Length (yrs, 0.5 = <1yr)",
                                      0.0, 11.0, float(D["emp_length"]), 0.5)
    with c5:
        home_ownership = st.selectbox("Home Ownership",
            ["RENT","MORTGAGE","OWN","OTHER","NONE"],
            index=["RENT","MORTGAGE","OWN","OTHER","NONE"].index(D["home_ownership"]))
        dti = st.number_input("Debt-to-Income Ratio", 0.0, 60.0,
                               float(D["dti"]), 0.1)
    with c6:
        verification_status = st.selectbox("Verification Status",
            ["Not Verified","Source Verified","Verified"],
            index=["Not Verified","Source Verified","Verified"].index(
                D["verification_status"]))
        pub_rec = st.number_input("Public Records", 0.0, 10.0,
                                   float(D["pub_rec"]), 1.0)

    # ── Section 3: Credit history ─────────────────────────────────────────────
    st.markdown('<div class="section-head">Credit History</div>', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        revol_bal  = st.number_input("Revolving Balance ($)", 0.0, 100000.0,
                                      float(D["revol_bal"]), 100.0)
        revol_util = st.number_input("Revolving Utilisation (%)", 0.0, 100.0,
                                      float(D["revol_util"]), 0.1)
    with c8:
        open_acc   = st.number_input("Open Credit Lines", 0.0, 50.0,
                                      float(D["open_acc"]), 1.0)
        total_acc  = st.number_input("Total Credit Lines", 0.0, 80.0,
                                      float(D["total_acc"]), 1.0)
    with c9:
        delinq_2yrs     = st.number_input("Delinquencies (2 yrs)", 0.0, 15.0,
                                           float(D["delinq_2yrs"]), 1.0)
        inq_last_6mths  = st.number_input("Credit Inquiries (6 mths)", 0.0, 20.0,
                                           float(D["inq_last_6mths"]), 1.0)

    # ── Section 4: Payment history ────────────────────────────────────────────
    st.markdown('<div class="section-head">Payment History</div>', unsafe_allow_html=True)
    c10, c11, c12 = st.columns(3)
    with c10:
        total_rec_prncp = st.number_input("Principal Received ($)", 0.0, 40000.0,
                                           float(D["total_rec_prncp"]), 10.0)
        total_rec_int   = st.number_input("Interest Received ($)", 0.0, 15000.0,
                                           float(D["total_rec_int"]), 10.0)
    with c11:
        last_pymnt_amnt = st.number_input("Last Payment Amount ($)", 0.0, 10000.0,
                                           float(D["last_pymnt_amnt"]), 1.0)
    with c12:
        st.caption("Payment amount and history are used in repayment ratio calculation.")

    # ── Section 5: Date features ──────────────────────────────────────────────
    st.markdown('<div class="section-head">Date Information</div>', unsafe_allow_html=True)
    with st.expander("Expand date fields (month/year)", expanded=False):
        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            ecl_m = st.number_input("Earliest Credit Line Month", 1, 12, D["earliest_cr_line_M"])
            ecl_y = st.number_input("Earliest Credit Line Year",  1950, 2025, D["earliest_cr_line_Y"])
        with dc2:
            isd_m = st.number_input("Issue Date Month", 1, 12, D["issue_d_M"])
            isd_y = st.number_input("Issue Date Year",  2000, 2025, D["issue_d_Y"])
        with dc3:
            lpd_m = st.number_input("Last Payment Month", 1, 12, D["last_pymnt_d_M"])
            lpd_y = st.number_input("Last Payment Year",  2000, 2025, D["last_pymnt_d_Y"])
        with dc4:
            lcd_m = st.number_input("Last Credit Pull Month", 1, 12, D["last_credit_pull_d_M"])
            lcd_y = st.number_input("Last Credit Pull Year",  2000, 2025, D["last_credit_pull_d_Y"])

    submitted = st.form_submit_button("🔍 Assess Default Risk",
                                       use_container_width=True,
                                       type="primary")


# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "term": int(term), "int_rate": int_rate,
        "emp_length": emp_length, "annual_inc": annual_inc,
        "dti": dti, "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": inq_last_6mths, "open_acc": open_acc,
        "pub_rec": pub_rec, "revol_bal": revol_bal,
        "revol_util": revol_util, "total_acc": total_acc,
        "total_rec_prncp": total_rec_prncp, "total_rec_int": total_rec_int,
        "last_pymnt_amnt": last_pymnt_amnt, "loan_amnt": loan_amnt,
        "installment": installment, "total_pymnt": total_pymnt,
        "earliest_cr_line_M": int(ecl_m), "earliest_cr_line_Y": int(ecl_y),
        "issue_d_M": int(isd_m), "issue_d_Y": int(isd_y),
        "last_pymnt_d_M": int(lpd_m), "last_pymnt_d_Y": int(lpd_y),
        "last_credit_pull_d_M": int(lcd_m), "last_credit_pull_d_Y": int(lcd_y),
        "home_ownership": home_ownership,
        "verification_status": verification_status,
        "purpose": purpose
    }

    with st.spinner("Analysing application..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)

            if resp.status_code == 200:
                r    = resp.json()
                prob = r["default_probability"]
                lbl  = r["prediction_label"]
                risk = r["risk_level"]
                shap = r["top_shap_features"]
                loss = r["expected_loss_usd"]

                st.markdown("---")
                st.subheader("📊 Assessment Results")

                # ── Top 3 metric cards
                m1, m2, m3 = st.columns(3)
                with m1:
                    css = ("result-default" if lbl == "DEFAULT"
                           else "result-medium" if risk == "MEDIUM"
                           else "result-safe")
                    icon = "🔴" if lbl=="DEFAULT" else ("🟡" if risk=="MEDIUM" else "🟢")
                    st.markdown(
                        f'<div class="{css}"><h2>{icon} {lbl}</h2>'
                        f'<p>Risk: <strong>{risk}</strong></p></div>',
                        unsafe_allow_html=True
                    )
                with m2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.metric("Default Probability", f"{prob:.2%}")
                    bar_color = ("#c0392b" if prob > 0.6
                                 else "#f39c12" if prob > 0.3
                                 else "#27ae60")
                    st.markdown(
                        f'<div class="prob-bar-wrap">'
                        f'<div style="background:{bar_color};width:{prob*100:.0f}%;'
                        f'height:14px;border-radius:6px"></div></div>',
                        unsafe_allow_html=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                with m3:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.metric("Expected Loss if Approved", f"${loss:,.0f}")
                    st.caption(f"Threshold: {r['threshold_used']} "
                               f"(cost-optimised for this dataset)")
                    st.markdown('</div>', unsafe_allow_html=True)

                # ── SHAP explanation chart
                if shap:
                    st.markdown("---")
                    st.subheader("🔬 Why this prediction? (SHAP Feature Contributions)")
                    st.caption(
                        "Red bars push toward DEFAULT. Blue bars push toward NON-DEFAULT. "
                        "The magnitude shows how strongly each feature influenced this decision."
                    )

                    sorted_shap = sorted(shap.items(),
                                         key=lambda x: abs(x[1]),
                                         reverse=True)[:10]
                    feats  = [f for f, _ in sorted_shap]
                    values = [v for _, v in sorted_shap]

                    fig, ax = plt.subplots(figsize=(9, 4))
                    colors  = ["#c0392b" if v > 0 else "#2980b9" for v in values]
                    ax.barh(feats[::-1], values[::-1],
                            color=colors[::-1], height=0.6, edgecolor="white")
                    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
                    ax.set_xlabel("SHAP Value (impact on default probability)",
                                  fontsize=11)
                    ax.set_title("Top 10 Feature Contributions", fontsize=13,
                                 fontweight="bold")
                    ax.spines[["top","right"]].set_visible(False)
                    red_patch  = mpatches.Patch(color="#c0392b", label="Increases default risk")
                    blue_patch = mpatches.Patch(color="#2980b9", label="Decreases default risk")
                    ax.legend(handles=[red_patch, blue_patch],
                              fontsize=9, loc="lower right")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # Table view
                    with st.expander("View full SHAP values table"):
                        shap_df = pd.DataFrame(
                            sorted_shap, columns=["Feature", "SHAP Value"]
                        )
                        shap_df["Direction"] = shap_df["SHAP Value"].apply(
                            lambda x: "↑ Default" if x > 0 else "↓ Non-default"
                        )
                        st.dataframe(shap_df, use_container_width=True)

                # ── Raw API response
                with st.expander("🔧 Raw API JSON Response"):
                    st.json(r)

            else:
                detail = resp.json().get("detail", "Unknown error")
                st.error(f"API error {resp.status_code}: {detail}")

        except requests.exceptions.ConnectionError:
            st.error(
                f"**Cannot connect to API** at `{API_URL}`\n\n"
                "Make sure the backend is running:\n"
                "```bash\nuvicorn api.main:app --host 0.0.0.0 --port 8000\n```"
            )
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── Footer
st.markdown("---")
st.caption("Loan Default Predictor · XGBoost + SHAP · FastAPI + Streamlit")