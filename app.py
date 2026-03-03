import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from utils import (
    load_artifacts,
    preprocess_input,
    preprocess_batch,
    risk_label,
    bar_chart,
    radar_chart,
    shap_waterfall_chart,
    create_pdf_report,
)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: Page Configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Heart Disease Risk Prediction",
    page_icon  = "❤️",
    layout     = "wide",   # Use the full screen width
)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: Custom CSS Styling
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Slightly warm background for the main area */
    .main { background-color: #FFF8F8 !important; }

    /* Sidebar background */
    [data-testid="stSidebar"] { background-color: #0D1B2A !important; }

    /* Make the risk % metric stand out in red */
    [data-testid="stMetricValue"] { color: #E05C5C !important; font-weight: 700; }

    /* Coloured card for the risk result */
    .risk-card {
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        font-size: 1.05rem;
    }
    .card-low    { background: #E8F5E9; border-left: 6px solid #4CAF50; }   /* green  */
    .card-medium { background: #FFF8E1; border-left: 6px solid #FFC107; }   /* yellow */
    .card-high   { background: #FFEBEE; border-left: 6px solid #E05C5C; }   /* red    */

    /* Orange disclaimer box */
    .disclaimer {
        background: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #5D4037;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: Load Model Files   
# ──────────────────────────────────────────────────────────────────────────────

try:
    model, scaler, features = load_artifacts()
except FileNotFoundError as error_message:
    st.error(str(error_message))
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: Session State
# ──────────────────────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5: App Header
# ──────────────────────────────────────────────────────────────────────────────

st.title("❤️ Heart Disease Risk Prediction")
st.caption("Graduation Project — AI-powered cardiovascular risk assessment tool")

st.markdown("""
<div class="disclaimer">
⚠️ <b>Medical Disclaimer:</b> This tool is for <b>educational purposes only</b> and does
<b>not</b> constitute medical advice. Always consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6: Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab_single, tab_batch, tab_history, tab_about = st.tabs([
    "🧑 Single Patient",
    "📋 Batch Prediction",
    "🗂 History",
    "ℹ️ About",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PATIENT
# ══════════════════════════════════════════════════════════════════════════════

with tab_single:

    st.sidebar.header("🧾 Patient Information")
    st.sidebar.markdown("---")

    age      = st.sidebar.slider("Age", 18,100, 55)
    sex_text = st.sidebar.selectbox("Sex", ["Female", "Male"])
    sex      = 0 if sex_text == "Female" else 1

    st.sidebar.markdown("##### 🩺 Clinical Measurements")
    resting_bp     = st.sidebar.slider("Resting Blood Pressure (mmHg)", 90,  190, 130)
    cholesterol    = st.sidebar.slider("Cholesterol (mg/dL)",           120, 350, 220)
    max_heart_rate = st.sidebar.slider("Max Heart Rate (bpm)",          90,  210, 150)
    bmi            = st.sidebar.slider("BMI",                           18.0, 40.0, 26.0)

    fbs_text = st.sidebar.selectbox("Fasting Blood Sugar", ["Low", "Normal", "High"])
    fasting_blood_sugar = 1 if fbs_text == "High" else 0

    st.sidebar.markdown("##### 🏃 Lifestyle Factors")
    smoking_text            = st.sidebar.selectbox("Smoking",         ["No", "Yes"])
    alcohol_text            = st.sidebar.selectbox("Alcohol Intake",  ["No", "Yes"])
    family_text             = st.sidebar.selectbox("Family History",  ["No", "Yes"])
    physical_activity_level = st.sidebar.slider("Physical Activity Level (1=Low, 5=High)", 1, 5, 3)
    stress_level            = st.sidebar.slider("Stress Level (1–10)",  1, 10, 5)
    sleep_hours             = st.sidebar.slider("Sleep Hours per Night", 4.0, 12.0, 7.0)

    smoking        = 0 if smoking_text == "No" else 1
    alcohol_intake = 0 if alcohol_text == "No" else 1
    family_history = 0 if family_text  == "No" else 1

    predict_btn = st.sidebar.button("🚀 Predict Risk", use_container_width=True)

    col_summary, col_result = st.columns([2, 1], gap="large")

    with col_summary:
        st.subheader("📌 Patient Summary")
        st.markdown(f"""
| Feature | Value |
|---|---|
| Age | {age} |
| Sex | {sex_text} |
| Resting BP | {resting_bp} mmHg |
| Cholesterol | {cholesterol} mg/dL |
| Max Heart Rate | {max_heart_rate} bpm |
| BMI | {bmi:.1f} |
| Fasting Blood Sugar | {fbs_text} |
| Smoking | {smoking_text} |
| Physical Activity | {physical_activity_level} / 5 |
| Alcohol Intake | {alcohol_text} |
| Family History | {family_text} |
| Stress Level | {stress_level} / 10 |
| Sleep Hours | {sleep_hours:.1f} h |
""")

    with col_result:
        st.subheader("📊 Prediction Result")

        if predict_btn:

            patient_data = {
                "age"                    : age,
                "sex"                    : sex,
                "resting_bp"             : resting_bp,
                "cholesterol"            : cholesterol,
                "max_heart_rate"         : max_heart_rate,
                "bmi"                    : bmi,
                "fasting_blood_sugar"    : fasting_blood_sugar,
                "smoking"                : smoking,
                "physical_activity_level": physical_activity_level,
                "alcohol_intake"         : alcohol_intake,
                "family_history"         : family_history,
                "stress_level"           : stress_level,
                "sleep_hours"            : sleep_hours,
            }

            input_df   = preprocess_input(patient_data, scaler, features)
            prediction = model.predict(input_df)[0]               # 0 or 1

            risk_pct = model.predict_proba(input_df)[0][1] * 100

            label, level = risk_label(risk_pct)

            css_class = {"low": "card-low", "medium": "card-medium", "high": "card-high"}[level]
            st.markdown(
                f'<div class="risk-card {css_class}"><b style="color: #0D1B2A;">{label}</b></div>',
                unsafe_allow_html=True,
            )

            st.metric("Heart Disease Risk", f"{risk_pct:.2f}%")
            st.progress(min(int(risk_pct), 100))
            st.write(f"**Prediction Class:** `{prediction}` (0 = No,  1 = Yes)")

            with st.expander("🔍 Why this prediction? (SHAP explanation)", expanded=False):
                shap_fig = shap_waterfall_chart(model, input_df)
                if shap_fig:
                    st.pyplot(shap_fig, use_container_width=True)
                else:
                    st.info("SHAP explanation is not available for this model type.")

            radar_data = {
                "BP"         : resting_bp / 190,
                "Cholesterol": cholesterol / 350,
                "BMI"        : bmi / 40,
                "Stress"     : stress_level / 10,
                "Smoking"    : float(smoking),
                "FBS High"   : float(fasting_blood_sugar),
            }
            radar_fig = radar_chart(radar_data)

            pdf_bytes = create_pdf_report(
                patient_data, risk_pct, label, prediction, radar_fig
            )

            st.download_button(
                label     = "📄 Download PDF Report",
                data      = pdf_bytes,
                file_name = "Heart_Disease_Prediction_Report.pdf",
                mime      = "application/pdf",
                use_container_width=True,
            )
            plt.close("all")

            st.session_state.history.append({
                "Age"        : age,
                "Sex"        : sex_text,
                "BP"         : resting_bp,
                "Cholesterol": cholesterol,
                "BMI"        : bmi,
                "Smoking"    : smoking_text,
                "Risk %"     : round(risk_pct, 2),
                "Risk Level" : label,
                "Prediction" : prediction,
            })

        else:
            st.info("👈 Fill in the sidebar and click **Predict Risk**")

    st.divider()
    st.subheader("📈 Patient Risk Visualisations")

    chart_left, chart_right = st.columns(2)

    with chart_left:
        bar_data = {
            "Blood Pressure" : resting_bp,
            "Cholesterol"    : cholesterol,
            "Max Heart Rate" : max_heart_rate,
            "BMI"            : bmi,
            "Stress Level"   : stress_level,
            "Sleep Hours"    : sleep_hours,
        }
        st.pyplot(bar_chart(bar_data), use_container_width=True)

    with chart_right:   
        radar_data = {
            "BP"         : resting_bp / 190,
            "Cholesterol": cholesterol / 350,
            "BMI"        : bmi / 40,
            "Stress"     : stress_level / 10,
            "Smoking"    : float(smoking),
            "FBS High"   : float(fasting_blood_sugar),
        }
        st.pyplot(radar_chart(radar_data), use_container_width=True)

    plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.subheader("📋 Batch Prediction")
    st.write(
        "Upload a CSV file with one patient per row. "
        "Required columns match the single-patient form above."
    )

    template_columns = [
        "age", "sex", "resting_bp", "cholesterol", "max_heart_rate", "bmi",
        "fasting_blood_sugar", "smoking", "physical_activity_level",
        "alcohol_intake", "family_history", "stress_level", "sleep_hours",
    ]
    empty_template = pd.DataFrame(columns=template_columns)

    st.download_button(
        label     = "⬇️ Download CSV Template",
        data      = empty_template.to_csv(index=False),
        file_name = "batch_template.csv",
        mime      = "text/csv",
    )

    uploaded_file = st.file_uploader("Upload your patient CSV", type="csv")

    if uploaded_file is not None:
        try:
            batch_raw = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_raw)}** patients.")
            st.dataframe(batch_raw.head(), use_container_width=True)

            batch_processed = preprocess_batch(batch_raw, scaler, features)
            predictions     = model.predict(batch_processed)
            probabilities   = model.predict_proba(batch_processed)[:, 1] * 100

            result_df                 = batch_raw.copy()
            result_df["Risk_%"]       = np.round(probabilities, 2)
            result_df["Risk_Level"]   = [risk_label(p)[0] for p in probabilities]
            result_df["Prediction"]   = predictions

            st.subheader("Results")
            st.dataframe(result_df, use_container_width=True)

            risk_counts = pd.Series(result_df["Risk_Level"]).value_counts()
            fig_batch, ax_batch = plt.subplots(figsize=(5, 3))
            risk_counts.plot(
                kind="bar", color=["#4CAF50", "#FFC107", "#E05C5C"],
                ax=ax_batch, edgecolor="white", rot=0
            )
            ax_batch.set_title("Risk Level Distribution (Batch)")
            ax_batch.set_ylabel("Number of Patients")
            ax_batch.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_batch, use_container_width=True)
            plt.close("all")

            st.download_button(
                label     = "⬇️ Download Results CSV",
                data      = result_df.to_csv(index=False).encode("utf-8"),
                file_name = "batch_predictions.csv",
                mime      = "text/csv",
            )

        except Exception as error:
            st.error(f"Error processing file: {error}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.subheader("🗂 Prediction History")

    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        if len(history_df) > 1:
            fig_trend, ax_trend = plt.subplots(figsize=(7, 3))
            prediction_numbers = history_df.index + 1

            ax_trend.plot(prediction_numbers, history_df["Risk %"],
                          marker="o", color="#E05C5C", linewidth=2)
            ax_trend.fill_between(prediction_numbers, history_df["Risk %"],
                                  alpha=0.1, color="#E05C5C")

            ax_trend.axhline(35, linestyle="--", color="#FFC107", linewidth=1,
                             label="Medium threshold (35%)")
            ax_trend.axhline(70, linestyle="--", color="#E05C5C", linewidth=1,
                             label="High threshold (70%)")

            ax_trend.set_title("Risk % Across Predictions")
            ax_trend.set_xlabel("Prediction Number")
            ax_trend.set_ylabel("Risk %")
            ax_trend.legend(fontsize=8)
            ax_trend.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_trend, use_container_width=True)
            plt.close("all")

        st.download_button(
            label     = "⬇️ Export History CSV",
            data      = history_df.to_csv(index=False).encode("utf-8"),
            file_name = "prediction_history.csv",
            mime      = "text/csv",
        )

        if st.button("🧹 Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No predictions yet. Go to the **Single Patient** tab to make your first prediction.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
### Heart Disease Risk Prediction — Graduation Project

This app uses a machine-learning model trained on clinical and lifestyle data
to estimate the probability that a patient has heart disease.

---

#### 🔬 Model Pipeline

| Step | Detail |
|---|---|
| **Candidates** | Random Forest, Gradient Boosting, XGBoost, Logistic Regression |
| **Selection** | 5-fold stratified cross-validation (ROC-AUC scoring) |
| **Tuning** | GridSearchCV on the winning model |
| **Class balance** | SMOTE oversampling applied to training data |
| **Calibration** | Isotonic regression via CalibratedClassifierCV |
| **Explainability** | SHAP TreeExplainer waterfall chart per patient |

---

#### 📐 Input Features

`age`, `sex`, `resting_bp`, `cholesterol`, `max_heart_rate`, `bmi`,
`fasting_blood_sugar`, `smoking`, `physical_activity_level`,
`alcohol_intake`, `family_history`, `stress_level`, `sleep_hours`

---

#### ⚠️ Disclaimer

This tool is developed for **educational and research purposes only**.
It does **not** replace professional medical advice, diagnosis, or treatment.
Always consult a qualified physician.

---

#### 🗂 Project Folder Structure

```
heart_disease_project/
├── data/                  ← Excel dataset
├── models/                ← Saved .pkl files (created by train.py)
├── evaluation/            ← Charts created by evaluate.py
├── src/
│   ├── train.py           ← Trains and saves the model
│   ├── evaluate.py        ← Evaluates the model and creates charts
│   └── utils.py           ← Shared helper functions
├── app.py                 ← This Streamlit app
└── requirements.txt       ← All required Python packages
```

#### ▶️ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (only needed once)
python src/train.py

# 3. (Optional) Generate evaluation charts
python src/evaluate.py

# 4. Launch the app
streamlit run app.py
```
""")