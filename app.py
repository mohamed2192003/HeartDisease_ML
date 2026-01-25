import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------------------
# Load Model Files
# -------------------------------
model = joblib.load("best_heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features_columns.pkl")

# -------------------------------
# Title
# -------------------------------
st.title("❤️ Heart Disease Risk Prediction (Graduation Project)")
st.write("Enter patient details in the sidebar, then click **Predict Risk**.")

# -------------------------------
# Session State (History Table)
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🧾 Patient Information")

age = st.sidebar.slider("Age", 30, 85, 55)

sex_text = st.sidebar.selectbox("Sex", ["Female", "Male"])
sex = 0 if sex_text == "Female" else 1

resting_bp = st.sidebar.slider("Resting Blood Pressure", 90, 190, 130)
cholesterol = st.sidebar.slider("Cholesterol", 120, 350, 220)
max_heart_rate = st.sidebar.slider("Max Heart Rate", 90, 210, 150)

bmi = st.sidebar.slider("BMI", 18.0, 40.0, 26.0)

fbs_text = st.sidebar.selectbox("Fasting Blood Sugar", ["Low", "Normal", "High"])
fasting_blood_sugar = 1 if fbs_text == "High" else 0

smoking_text = st.sidebar.selectbox("Smoking", ["No", "Yes"])
smoking = 0 if smoking_text == "No" else 1

physical_activity_level = st.sidebar.slider("Physical Activity Level (1=Low, 5=High)", 1, 5, 3)

alcohol_text = st.sidebar.selectbox("Alcohol Intake", ["No", "Yes"])
alcohol_intake = 0 if alcohol_text == "No" else 1

family_text = st.sidebar.selectbox("Family History", ["No", "Yes"])
family_history = 0 if family_text == "No" else 1

stress_level = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
sleep_hours = st.sidebar.slider("Sleep Hours", 4.0, 9.0, 7.0)

predict_btn = st.sidebar.button("🚀 Predict Risk")


# -------------------------------
# Helper: Radar Chart
# -------------------------------
def radar_chart(values_dict):
    labels = list(values_dict.keys())
    values = list(values_dict.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Patient Risk Factors Radar Chart")

    return fig


# -------------------------------
# Helper: Create PDF Report
# -------------------------------
def create_pdf_report(patient_data, risk_percent, risk_level, prediction_class):
    pdf_file = "prediction_report.pdf"

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_file, pagesize=A4)

    story = []
    story.append(Paragraph("Heart Disease Risk Prediction Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Risk Level:</b> {risk_level}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Risk Percentage:</b> {risk_percent:.2f}%", styles["BodyText"]))
    story.append(Paragraph(f"<b>Prediction Class:</b> {prediction_class} (0=No, 1=Yes)", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Patient Input Data:", styles["Heading2"]))
    table_data = [["Feature", "Value"]] + [[k, str(v)] for k, v in patient_data.items()]

    table = Table(table_data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))

    story.append(table)
    doc.build(story)

    return pdf_file


# -------------------------------
# Main Layout
# -------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📌 Patient Summary")
    st.write(
        f"""
        - **Age:** {age}  
        - **Sex:** {sex_text}  
        - **Resting BP:** {resting_bp}  
        - **Cholesterol:** {cholesterol}  
        - **Max Heart Rate:** {max_heart_rate}  
        - **BMI:** {bmi}  
        - **Fasting Blood Sugar:** {fbs_text}  
        - **Smoking:** {smoking_text}  
        - **Physical Activity Level:** {physical_activity_level}  
        - **Alcohol Intake:** {alcohol_text}  
        - **Family History:** {family_text}  
        - **Stress Level:** {stress_level}  
        - **Sleep Hours:** {sleep_hours}  
        """
    )

with col2:
    st.subheader("📊 Prediction Result")

    if predict_btn:

        # Prepare Input
        patient_data = {
            "age": age,
            "sex": sex,
            "resting_bp": resting_bp,
            "cholesterol": cholesterol,
            "max_heart_rate": max_heart_rate,
            "bmi": bmi,
            "fasting_blood_sugar": fasting_blood_sugar,
            "smoking": smoking,
            "physical_activity_level": physical_activity_level,
            "alcohol_intake": alcohol_intake,
            "family_history": family_history,
            "stress_level": stress_level,
            "sleep_hours": sleep_hours
        }

        input_df = pd.DataFrame([patient_data])
        input_df = pd.get_dummies(input_df, drop_first=True)

        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[features]

        numeric_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
            risk_percent = probability * 100
        else:
            risk_percent = 0

        # Risk Level Colors
        if risk_percent < 35:
            risk_level = "LOW RISK ✅"
            st.success(risk_level)
        elif risk_percent < 70:
            risk_level = "MEDIUM RISK ⚠️"
            st.warning(risk_level)
        else:
            risk_level = "HIGH RISK ❌"
            st.error(risk_level)

        st.metric("Heart Disease Risk", f"{risk_percent:.2f}%")
        st.progress(int(risk_percent))

        st.write(f"**Prediction Class:** `{prediction}` (0=No, 1=Yes)")

        # Save to History
        st.session_state.history.append({
            "Age": age,
            "Sex": sex_text,
            "BP": resting_bp,
            "Cholesterol": cholesterol,
            "BMI": bmi,
            "Smoking": smoking_text,
            "Risk %": round(risk_percent, 2),
            "Risk Level": risk_level
        })

        # PDF Download
        pdf_file = create_pdf_report(patient_data, risk_percent, risk_level, prediction)
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="📄 Download Prediction Report (PDF)",
                data=f,
                file_name="Heart_Disease_Prediction_Report.pdf",
                mime="application/pdf"
            )

    else:
        st.info("👈 Enter values in the sidebar and click **Predict Risk**")

st.divider()
st.subheader("📈 Charts")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.write("### Bar Chart (Main Features)")
    bar_data = {
        "BP": resting_bp,
        "Cholesterol": cholesterol,
        "BMI": bmi,
        "Stress": stress_level,
        "Sleep": sleep_hours
    }
    fig1 = plt.figure(figsize=(5, 3))
    plt.bar(bar_data.keys(), bar_data.values())
    plt.title("Patient Health Factors")
    plt.ylabel("Value")
    st.pyplot(fig1)

with chart_col2:
    st.write("### Radar Chart (Risk Profile)")
    radar_data = {
        "BP": resting_bp / 190,
        "Cholesterol": cholesterol / 350,
        "BMI": bmi / 40,
        "Stress": stress_level / 10,
        "Smoking": smoking,
        "FBS High": fasting_blood_sugar
    }
    fig2 = radar_chart(radar_data)
    st.pyplot(fig2)


st.divider()
st.subheader("🗂 Prediction History")

if len(st.session_state.history) > 0:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    if st.button("🧹 Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No predictions yet. Make your first prediction!") 