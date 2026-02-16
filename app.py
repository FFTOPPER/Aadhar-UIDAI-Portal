import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# =================================================
# Page configuration
# =================================================
st.set_page_config(
    page_title="Aadhaar Analytics Dashboard",
    layout="wide"
)

# =================================================
# Custom CSS — ONLY to fix white sidebar
# =================================================
st.markdown(
    """
    <style>
    /* Sidebar width */
    section[data-testid="stSidebar"] {
        width: 320px !important;
        background-color: #0b0f14 !important;
    }

    /* Sidebar inner container */
    section[data-testid="stSidebar"] > div {
        background-color: #0b0f14 !important;
        padding: 15px;
    }

    /* Sidebar title */
    section[data-testid="stSidebar"] h2 {
        color: #ffffff;
    }

    /* Navigation blocks */
    div[role="radiogroup"] > label {
        display: block;
        background-color: #151a21;
        color: #ffffff;
        border: 2px solid #2a2f3a;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
    }

    /* Hover effect */
    div[role="radiogroup"] > label:hover {
        background-color: #1f2630;
        border-color: #3b82f6;
    }

    /* Selected item */
    div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #1e3a8a;
        border-color: #60a5fa;
        font-weight: 600;
    }

    /* Hide radio circle */
    div[role="radiogroup"] input {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =================================================
# Load Models and Data
# =================================================
ems_model = joblib.load("ems_model.pkl")
ems_features = joblib.load("ems_features.pkl")

dup_kmeans = joblib.load("dup_kmeans_model.pkl")
dup_scaler = joblib.load("dup_scaler.pkl")
dup_features = joblib.load("dup_features.pkl")
dup_lr_models = joblib.load("dup_lr_models.pkl")
dup_monthly_data = joblib.load("dup_monthly_data.pkl")

ltp_lr_model = joblib.load("ltp_lr_model.pkl")
ltp_model_df = joblib.load("ltp_model_df.pkl")

adr_kmeans = joblib.load("adr_kmeans_model.pkl")
adr_scaler = joblib.load("adr_scaler.pkl")
adr_cluster_labels = joblib.load("adr_cluster_labels.pkl")

# =================================================
# Sidebar Navigation
# =================================================
st.sidebar.markdown(
    "<h2 style='text-align:center;'>Dashboard Modules</h2>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    [
        "Home",
        "EMS – Enrolment Momentum",
        "DUP – Demographic Update Pressure",
        "LTP – Lifecycle Transition Predictor",
        "ADR – Aadhaar Dormancy Risk"
    ]
)

# =================================================
# Header
# =================================================
st.markdown(
    """
    <h1 style="text-align:center;">Aadhaar Analytics Dashboard</h1>
    <p style="text-align:center; font-size:17px;">
    Enrolment Momentum and Update Dynamics – Decision Support System
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# =================================================
# HOME PAGE
# =================================================
if page == "Home":
    st.subheader("Welcome to the Aadhaar Analytics Dashboard")

    st.markdown(
        """
        This dashboard is designed as an integrated **decision-support system**
        using anonymised Aadhaar datasets released by UIDAI.

        It combines enrolment analytics, update workload assessment,
        lifecycle-driven forecasting, and post-enrolment engagement analysis
        to support informed, proactive decision-making.
        """
    )

    st.subheader("Analytical Modules Overview")

    st.markdown(
        """
        **EMS – Enrolment Momentum Score**  
        Analyses recent enrolment patterns and identifies whether enrolment
        activity is increasing, declining, or uncertain.

        **DUP – Demographic Update Pressure**  
        Identifies regions under higher demographic update workload and
        forecasts near-term update demand.

        **LTP – Lifecycle Transition Predictor**  
        Estimates future biometric update demand driven by predictable
        age-based lifecycle transitions.

        **ADR – Aadhaar Dormancy Risk**  
        Evaluates post-enrolment engagement to identify Aadhaar records
        at risk of dormancy due to low interaction.
        """
    )

    st.info("Select a module from the left panel to view detailed analysis.")

# =================================================
# EMS MODULE
# =================================================
elif page == "EMS – Enrolment Momentum":
    st.subheader("Enrolment Momentum Score (EMS)")

    st.markdown(
        "Enter recent enrolment values to assess whether enrolment activity "
        "is increasing, declining, or uncertain."
    )

    with st.form("ems_input_form"):
        region = st.text_input("Region Name (for reference)", "Tamil Nadu")

        col1, col2, col3 = st.columns(3)
        with col1:
            lag_1 = st.number_input("Last Month Enrolment", min_value=0, value=30000, step=1000)
        with col2:
            lag_2 = st.number_input("2 Months Ago Enrolment", min_value=0, value=28000, step=1000)
        with col3:
            lag_3 = st.number_input("3 Months Ago Enrolment", min_value=0, value=26000, step=1000)

        predict_btn = st.form_submit_button("Predict Enrolment Momentum")

    st.caption("Predictions are generated based on enrolment patterns, not the region name.")

    if predict_btn:
        input_df = pd.DataFrame([[lag_1, lag_2, lag_3]], columns=ems_features)
        probabilities = ems_model.predict_proba(input_df)[0]
        predicted_class = ems_model.predict(input_df)[0]

        max_prob = max(probabilities)

        if max_prob >= 0.85:
            confidence = "High Confidence"
        elif max_prob >= 0.65:
            confidence = "Medium Confidence"
        else:
            confidence = "Low Confidence"

        if max_prob < 0.60:
            final_decision = "Uncertain"
        else:
            final_decision = ["Declining", "Stable", "Increasing"][predicted_class]

        colA, colB = st.columns([2, 1])

        with colA:
            st.subheader("Prediction Outcome")

            if final_decision == "Increasing":
                st.success("Enrolment Trend: Increasing")
            elif final_decision == "Declining":
                st.error("Enrolment Trend: Declining")
            else:
                st.warning("Enrolment Trend: Uncertain")

            st.write(f"Region: {region}")
            st.write(f"Confidence Level: {confidence}")

            fig, ax = plt.subplots()
            ax.bar(["Decline", "Stable", "Increase"], probabilities)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            st.pyplot(fig)

        with colB:
            st.subheader("Decision Guidance")

            if final_decision == "Increasing" and confidence == "High Confidence":
                st.info("Recommended Action:\nContinue current enrolment strategy.")
            elif final_decision == "Declining" and confidence == "High Confidence":
                st.warning(
                    "Recommended Action:\nConsider targeted outreach and resource allocation."
                )
            else:
                st.write("Recommended Action:\nMonitor enrolment trends before taking action.")

            st.subheader("Model Interpretation")
            st.write(
                "- Recent enrolment history has the strongest influence\n"
                "- Short-term fluctuations are handled conservatively\n"
                "- Low-confidence cases are explicitly marked as uncertain"
            )

# =================================================
# DUP MODULE
# =================================================
elif page == "DUP – Demographic Update Pressure":
    st.subheader("Demographic Update Pressure Analysis")

    st.markdown(
        "This module analyses demographic update activity to identify regions "
        "under higher update pressure and forecast near-term update demand."
    )

    cluster_map = {0: "Low Pressure", 1: "Moderate Pressure", 2: "High Pressure"}
    dup_features["Pressure Zone"] = dup_features["pressure_cluster"].map(cluster_map)

    st.dataframe(
        dup_features[["state", "dup_mean", "avg_update_load", "Pressure Zone"]]
        .sort_values("dup_mean", ascending=False),
        use_container_width=True
    )

    st.caption(
        "Pressure zones are derived using clustering on historical demographic update patterns."
    )

    st.subheader("Forecast of Demographic Update Demand")

    selected_state = st.selectbox("Select State", sorted(dup_lr_models.keys()))

    state_data = dup_monthly_data[
        dup_monthly_data["state"] == selected_state
    ].dropna().reset_index(drop=True)

    state_data["t"] = np.arange(len(state_data))
    lr_model = dup_lr_models[selected_state]

    future_t = np.arange(len(state_data), len(state_data) + 6).reshape(-1, 1)
    future_predictions = lr_model.predict(future_t)

    fig, ax = plt.subplots()
    ax.plot(state_data["t"], state_data["total_demo_updates"], label="Historical")
    ax.plot(future_t.flatten(), future_predictions, linestyle="--", label="Forecast")
    ax.set_title(f"Update Demand Forecast – {selected_state}")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Demographic Updates")
    ax.legend()
    st.pyplot(fig)

    st.caption(
        "The chart shows historical demographic update volumes and the projected near-term demand, "
        "which can support proactive planning of update infrastructure and staffing."
    )

    st.caption(
        "Forecasts are generated using state-specific regression models trained on historical data."
    )

# =================================================
# LTP MODULE
# =================================================
elif page == "LTP – Lifecycle Transition Predictor":
    st.subheader("Lifecycle Transition Predictor (LTP)")

    st.markdown(
        "This module estimates future biometric update demand caused by age-based "
        "lifecycle transitions, particularly individuals moving into adult biometric requirements."
    )

    selected_state = st.selectbox("Select State", sorted(ltp_model_df["state"].unique()))
    horizon = st.radio("Forecast Horizon", options=[6, 12], horizontal=True)

    state_df = ltp_model_df[
        ltp_model_df["state"] == selected_state
    ].reset_index(drop=True)

    if len(state_df) < 3:
        st.warning("Insufficient historical data to generate a reliable lifecycle forecast.")
    else:
        last_t = state_df["t"].max()
        avg_age_5_17 = state_df["age_5_17"].mean()

        future_X = pd.DataFrame({
            "age_5_17": [avg_age_5_17] * horizon,
            "t": range(last_t + 1, last_t + 1 + horizon)
        })

        future_predictions = ltp_lr_model.predict(future_X)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Biometric Update Demand Forecast")

            fig, ax = plt.subplots()
            ax.plot(state_df["t"], state_df["bio_age_17_"], label="Historical Updates")
            ax.plot(future_X["t"], future_predictions, linestyle="--", label="Projected Demand")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Biometric Updates (17+)")
            ax.set_title(f"{selected_state} – Lifecycle-Based Forecast")
            ax.legend()
            st.pyplot(fig)

            st.caption(
                "The forecast reflects expected biometric update demand driven by "
                "age-based lifecycle transitions."
            )

        with col2:
            st.subheader("Forecast Summary")

            st.metric("Expected Updates (Next Period)", f"{int(future_predictions.sum())}")
            st.metric("Average Monthly Updates", f"{int(future_predictions.mean())}")

            st.subheader("Planning Insight")
            st.write(
                "The projected demand provides an early indication of biometric update "
                "workload arising from population cohorts transitioning into adulthood. "
                "This can support advance planning of update infrastructure and staffing."
            )

# =================================================
# ADR MODULE
# =================================================
elif page == "ADR – Aadhaar Dormancy Risk":
    st.subheader("Aadhaar Dormancy Risk (ADR) Analysis")

    st.markdown(
        "This module evaluates post-enrolment engagement by analysing demographic "
        "and biometric update activity relative to enrolment volume."
    )

    col1, col2 = st.columns(2)
    with col1:
        der = st.number_input("Demographic Engagement Ratio (DER)", min_value=0.0, value=0.01, step=0.01)
    with col2:
        ber = st.number_input("Biometric Engagement Ratio (BER)", min_value=0.0, value=0.01, step=0.01)

    if st.button("Assess Dormancy Risk"):
        adr_score = 1 - (0.5 * der + 0.5 * ber)
        X = np.array([[adr_score, der, ber]])
        X_scaled = adr_scaler.transform(X)
        cluster = adr_kmeans.predict(X_scaled)[0]
        label = adr_cluster_labels.get(cluster, "Unknown")

        st.subheader("Dormancy Risk Assessment")

        if "High" in label:
            st.error(f"Dormancy Risk Level: {label}")
        elif "Moderate" in label:
            st.warning(f"Dormancy Risk Level: {label}")
        else:
            st.success(f"Dormancy Risk Level: {label}")

        st.caption(
            "Dormancy risk is inferred using unsupervised clustering on engagement ratios. "
            "Higher risk indicates lower post-enrolment interaction."
        )

# =================================================
# Footer
# =================================================
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px;">
    UIDAI Data Hackathon 2026 | EMS, DUP, LTP & ADR – Integrated Decision Support Dashboard
    </p>
    """,
    unsafe_allow_html=True
)
