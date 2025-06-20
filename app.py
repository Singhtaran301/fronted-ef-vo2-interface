import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import joblib
import altair as alt
import time
import os
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent import AgentExecutor

load_dotenv()
# Load Models
ef_model = joblib.load("catboost_ef_model.pkl")
vo2_model = joblib.load("vo2_model_rf.pkl")

# Page Setup
st.set_page_config(page_title="Heart Digital Twin", layout="wide")
st.title("Digital Twin Dashboard – Heart EF & VO₂")

# Sidebar Inputs
st.sidebar.header("Enter Patient Vitals")
hr = st.sidebar.number_input("Heart Rate (bpm)", value=85)
sbp = st.sidebar.number_input("Systolic BP (mmHg)", value=120)
dbp = st.sidebar.number_input("Diastolic BP (mmHg)", value=80)
spo2 = st.sidebar.number_input("SpO₂ (%)", value=96)
hb = st.sidebar.number_input("Hemoglobin (g/dL)", value=14.0)
pao2 = st.sidebar.number_input("PaO₂ (mmHg)", value=85.0)

# Input DataFrame (initialize with user inputs)
input_df = pd.DataFrame([{
    "Heart Rate": hr,
    "Hemoglobin": hb,
    "PaO2": pao2,
    "SpO2": spo2,
    "Systolic BP": sbp,
    "Diastolic BP": dbp
}])

# === EF Feature Engineering ===
input_df['CaO2'] = hb * 1.34 * (spo2 / 100) + 0.0031 * pao2
input_df['O2_Delivery'] = input_df['CaO2'] * hr
input_df['Hb_HR'] = hb * hr
input_df['PaO2_div_SpO2'] = pao2 / spo2
input_df['HR_SpO2'] = hr * spo2
input_df['HR_Hb_SpO2'] = hr * hb * spo2
input_df['CaO2_per_HR'] = input_df['CaO2'] / hr
input_df['Hb_div_PaO2'] = hb / pao2
input_df['Hb_times_logHR'] = hb * np.log(hr + 1)

# === VO2 Feature Engineering ===
input_df['f_1'] = hb * spo2 * 1.34
input_df['f_2'] = sbp * hr
input_df['f_3'] = sbp - dbp
input_df['oxy_delivery'] = (hb * 1.34 * spo2) * (hr / 1000)
input_df['cv_stress'] = (sbp * hr) / 1000
input_df['pp_ratio'] = (sbp - dbp) / sbp
input_df['mean'] = (2 * dbp + sbp) / 3
input_df['O2 content diff'] = (spo2 - pao2) * hb * 1.34
input_df['oer'] = (spo2 - pao2) / spo2
input_df['shock_index'] = hr / (sbp + 1e-6)
input_df['CaO2'] = hb * 1.34 * spo2

# Columns expected by the models
ef_input_cols = [
    'Heart Rate', 'Hemoglobin', 'PaO2', 'SpO2', 'CaO2', 'O2_Delivery', 'Hb_HR', 'PaO2_div_SpO2',
    'HR_SpO2', 'HR_Hb_SpO2', 'CaO2_per_HR', 'Hb_div_PaO2', 'Hb_times_logHR'
]
vo2_input_cols = [
    'Diastolic BP', 'Heart Rate', 'Hemoglobin', 'PaO2', 'SpO2', 'Systolic BP', 'f_1', 'f_2', 'f_3',
    'oxy_delivery', 'cv_stress', 'pp_ratio', 'mean', 'O2 content diff', 'oer', 'shock_index', 'CaO2'
]

# Predict EF & VO2
input_df["EF_percent"] = ef_model.predict(input_df[ef_input_cols])
input_df["VO2_ml_per_min"] = vo2_model.predict(input_df[vo2_input_cols])

# Extract predicted values
ef = input_df["EF_percent"].iloc[0]
vo2 = input_df["VO2_ml_per_min"].iloc[0]

# Layout: Image + predictions
col1, col2 = st.columns([1, 2])
with col1:
    st.image("download.jpg", caption="Cardiac Contraction Pattern", use_container_width=True)

with col2:
    st.markdown("### Predicted EF and VO₂")
    st.dataframe(input_df[["EF_percent", "VO2_ml_per_min"]].style.format(precision=2))

    st.markdown("### EF/VO₂ Alerts")
    if ef < 40:
        st.error(f"❗ EF is {ef:.2f}% — possible heart failure (HFrEF).")
    elif ef > 75:
        st.warning(f"⚠️ EF is {ef:.2f}% — possible hyperdynamic function.")
    else:
        st.success(f"✅ EF is normal: {ef:.2f}%")

    if vo2 < 250:
        st.error(f"❗ VO₂ is low: {vo2:.2f} ml/min — possible poor oxygen delivery.")
    elif vo2 > 400:
        st.warning(f"⚠️ VO₂ is elevated: {vo2:.2f} ml/min — check for stress/exertion.")
    else:
        st.success(f"✅ VO₂ is within normal range: {vo2:.2f} ml/min")

# Altair bar chart of EF and VO2
chart_data = pd.DataFrame({
    "Metric": ["EF_percent", "VO2_ml_per_min"],
    "Value": [ef, vo2]
})

chart = alt.Chart(chart_data).mark_bar().encode(
    x=alt.X("Metric", sort=None),
    y="Value",
    color=alt.Color("Metric", legend=None),
    tooltip=["Metric", "Value"]
).properties(
    width=500,
    height=300,
    title="EF and VO₂ Comparison Chart"
)

st.altair_chart(chart)

st.markdown("### AI Interpretation via Groq")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY environment variable not set! Please set your API key.")
    st.stop()

start_time = time.time()

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

groq_llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama3-8b-8192",
    temperature=0.3
)

# Create simple prompt
query = f"""
You are a medical assistant helping patients understand their test results.

Please explain the following values in simple, clinical language:

- Ejection Fraction (EF): {ef:.2f}%  
- VO₂ (Oxygen Consumption): {vo2:.2f} ml/min  

For each value:
1. Briefly describe what it means in layman's terms.
2. Mention if the value is within the normal range.
3. Give a short interpretation of whether it suggests good, borderline, or concerning health status.

Keep the tone informative and reassuring.
"""



with st.spinner("AI analyzing your heart data..."):
    try:
        response = groq_llm.invoke([HumanMessage(content=query)])
        st.success("AI Analysis Complete:")
        st.write(response.content)
    except Exception as e:
        st.error(f"LLM error: {str(e)}")

end_time = time.time()
st.write(f"LLM Response Time: {end_time - start_time:.2f} seconds")

# Downloadable report button
st.download_button(
    "Download EF/VO₂ Report",
    input_df.to_csv(index=False),
    "heart_report.csv"
)
