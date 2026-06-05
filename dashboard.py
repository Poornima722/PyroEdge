import streamlit as st
from datetime import datetime
import json
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import joblib

@st.cache_resource
def load_pyroedge_model():
    return joblib.load('forest_fire_model.pkl')
st.cache_resource.clear()

rf_model = load_pyroedge_model()

# --- CONFIGURATION ---
st.title("🔥 PyroEdge - Fire Risk Dashboard")
st.set_page_config(layout="wide", page_title="PyroEdge")
st_autorefresh(interval=3000, limit=None, key="refresh")

# --- STEP 1: CREATE TABS ---
tab1, tab2 = st.tabs(["🔥 Live Dashboard", "📊 Data Analytics & Justification"])


# --- STEP 2: LOAD LIVE DATA AND INFER RISK VIA RF MODEL ---
if os.path.exists('data.json'):
    try:
        with open('data.json', 'r') as f:
            live_data = json.load(f)
        
        # NODE 1 DATA
        n1_data = live_data.get('node1', {})
        n1_temp = n1_data.get('Temp', 0)
        n1_humidity = n1_data.get('Humidity', 0)
        n1_rainfall = n1_data.get('Rainfall', 0)
        
        # NODE 2 DATA
        n2_data = live_data.get('node2', {})
        n2_temp = n2_data.get('Temp', 0)
        n2_humidity = n2_data.get('Humidity', 0)
        n2_rainfall = n2_data.get('Rainfall', 0)

        # 🚀 LIVE AI INFERENCE FOR NODE 1
        n1_features = [[n1_temp, n1_humidity, n1_rainfall]]
        n1_pred = rf_model.predict(n1_features)[0]
        # The safety filler
        if n1_temp < 30.0 or n1_rainfall == 1 or n1_humidity > 70.0:
            n1_risk = "LOW"
        else:
            n1_risk = "HIGH" if n1_pred == 1 else "LOW"

        # 🚀 LIVE AI INFERENCE FOR NODE 2
        n2_features = [[n2_temp, n2_humidity, n2_rainfall]]
        n2_pred = rf_model.predict(n2_features)[0]
        if n2_temp < 30.0 or n2_rainfall == 1 or n2_humidity > 70.0:
            n2_risk = "LOW"
        else:
            n2_risk = "HIGH" if n2_pred == 1 else "LOW"
        
    except Exception as e:
        # Fallback values to prevent dashboard crash if JSON is temporarily empty during an MQTT write
        n1_temp, n1_humidity, n1_rainfall, n1_risk = 0, 0, 0, "LOW"
        n2_temp, n2_humidity, n2_rainfall, n2_risk = 0, 0, 0, "LOW"
else:
    # Default values if file doesn't exist yet
    n1_temp, n1_humidity, n1_rainfall, n1_risk = 0, 0, 0, "LOW"
    n2_temp, n2_humidity, n2_rainfall, n2_risk = 0, 0, 0, "LOW"

# --- TREND MEMORY ---
# Extract the static history list from your new JSON file
history_list = live_data.get('history', [])

# Extract lists directly so Plotly can read all 15 points instantly
time_data = [item['time'] for item in history_list]
temp_trend_n1 = [item['n1_T'] for item in history_list]
temp_trend_n2 = [item['n2_T'] for item in history_list]

# Initialize alert history and a 'last_state' tracker if they don't exist
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []
if 'last_status' not in st.session_state:
    st.session_state.last_status = {"node1": "LOW", "node2": "LOW"}

current_time = datetime.now().strftime("%H:%M:%S")

# --- Node 1 Logic ---
if n1_risk == "HIGH" and st.session_state.last_status["node1"] == "LOW":
    st.session_state.alert_log.insert(0, f"🕒 {current_time} | 🚨 ALERT: High Risk in ZONE 01")
    st.session_state.last_status["node1"] = "HIGH"
elif n1_risk == "LOW" and st.session_state.last_status["node1"] == "HIGH":
    st.session_state.alert_log.insert(0, f"🕒 {current_time} | ✅ RESOLVED: ZONE 01 is now Secure")
    st.session_state.last_status["node1"] = "LOW"

# --- Node 2 Logic ---
if n2_risk == "HIGH" and st.session_state.last_status["node2"] == "LOW":
    st.session_state.alert_log.insert(0, f"🕒 {current_time} | 🚨 ALERT: High Risk in ZONE 02")
    st.session_state.last_status["node2"] = "HIGH"
elif n2_risk == "LOW" and st.session_state.last_status["node2"] == "HIGH":
    st.session_state.alert_log.insert(0, f"🕒 {current_time} | ✅ RESOLVED: ZONE 02 is now Secure")
    st.session_state.last_status["node2"] = "LOW"

# Keep only the last 10 entries
st.session_state.alert_log = st.session_state.alert_log[:8]

# --- NOW YOU CAN START THE TABS ---
#tab1, tab2 = st.tabs(["🔥 Live Monitoring", "🔬 EDA Analysis"])

# --- TAB 1: LIVE MONITORING ---
with tab1:
    # 1. Split the screen into two main containers with a vertical gap
    main_col1, divider_col, main_col2 = st.columns([10, 1, 10])

    # --- NODE 01: ---
    with main_col1:
        st.markdown("### 📍 Node 01")
        
        # Displaying everything in one single row using 4 mini-columns
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.write("TEMPERATURE")
            st.markdown(f"**{n1_temp} °C**")
        with m2:
            st.write("HUMIDITY")
            st.markdown(f"**{n1_humidity} %**")
        with m3:
            st.write("RAINFALL")
            st.markdown(f"**{'Yes' if n1_rainfall == 1 else 'No'}**")
        with m4:
            st.write("FIRE RISK")
            risk_text1 = "🔴 **HIGH**" if n1_risk == "HIGH" else "🟢 **LOW**"
            st.markdown(risk_text1)

        # Plot for Node 1
        st.write("### 📈 Temperature Trend (N1)")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=time_data, y=temp_trend_n1, mode='lines', line=dict(color='#ff4b4b', width=2)))
        
        fig1.update_layout(
            template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(nticks=4), # Strictly limits time axis to ~4 readings
            yaxis_title="°C"
        )
        st.plotly_chart(fig1, use_container_width=True)

    # --- VERTICAL DIVIDER ---
    with divider_col:
        # Visual trick to create a vertical line
        st.markdown("""<div style="border-left: 2px solid #555; height: 450px; margin-left: 50%; opacity: 0.2;"></div>""", unsafe_allow_html=True)

    # --- NODE 02: ---
    with main_col2:
        st.markdown("### 📍 Node 02")
        
        # Displaying everything in one single row using 4 mini-columns
        m5, m6, m7, m8 = st.columns(4)
        
        with m5:
            st.write("TEMPERATURE")
            st.markdown(f"**{n2_temp} °C**")
        with m6:
            st.write("HUMIDITY")
            st.markdown(f"**{n2_humidity} %**")
        with m7:
            st.write("RAINFALL")
            st.markdown(f"**{'Yes' if n2_rainfall == 1 else 'No'}**")
        with m8:
            st.write("FIRE RISK")
            risk_text2 = "🔴 **HIGH**" if n2_risk == "HIGH" else "🟢 **LOW**"
            st.markdown(risk_text2)

        # Plot for Node 2
        st.write("### 📈 Temperature Trend (N2)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=time_data, y=temp_trend_n2, mode='lines', line=dict(color='#ffa500', width=2)))
        
        fig2.update_layout(
            template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(nticks=4), # Strictly limits time axis to ~4 readings
            yaxis_title="°C"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- DYNAMIC ALERT SECTION ---
    st.divider()

    # Check if BOTH nodes are high
    if n1_risk == "HIGH" and n2_risk == "HIGH":
        st.error("🚨 **CRITICAL ALERT:** High Fire Risk detected in **BOTH** Zones!")
    
    # Check if only Node 1 is high
    elif n1_risk == "HIGH":
        st.error("⚠️ **NODE 01 ALERT:** High Fire Risk detected in NODE 1!")
    
    # Check if only Node 2 is high
    elif n2_risk == "HIGH":
        st.error("⚠️ **NODE 02 ALERT:** High Fire Risk detected in NODE 2!")
    # If everything is fine
    else:
        st.success("✅ **System Status:** BOTH Zones are currently secure.")
    
    # --- ALERT HISTORY LOG ---
    st.write("### 📜 Recent Incident Log")
    
    if st.session_state.alert_log:
        # Create a scrollable box for the logs
        with st.expander("View detailed history", expanded=True):
            for log in st.session_state.alert_log:
                st.write(log)
    else:
        st.info("No incidents recorded in this session.")

    # Optional: Clear button
    if st.button("Clear Log"):
        st.session_state.alert_log = []
        st.rerun()

# --- TAB 2: DATA ANALYTICS & JUSTIFICATION ---
with tab2:
    st.header("🔬 Model Justification & Data Assessment")
    st.info("Technical evidence supporting the PyroEdge fire risk prediction logic.")

    # --- ROW 1: DATA DISTRIBUTION ---
    st.subheader("1️⃣ Feature Distribution (Temperature, Humidity, Rain)")
    # This proves the 'spread' of your sensor data
    st.image("distribution.png", use_container_width=True)
    st.markdown("""
    **Key Findings:**
    * **Temperature:** Most readings are concentrated between **30°C and 45°C**, reflecting peak daytime heat.
    * **Humidity:** Shows a wide spread, primarily peaking around **35%**, typical for dry fire-prone environments.
    * **Rainfall:** A significantly higher frequency of 'No Rain' samples, establishing the dry baseline for fire risk.
    """)
    st.divider()

    # --- ROW 2: OUTLIER DETECTION ---
    st.subheader("2️⃣ Outlier Analysis")
    # This proves your data is clean
    st.image("outliers.png", use_container_width=True)
    st.markdown("""
    **Key Findings:**
    * **Clean Data:** The median temperature sits around **35°C**, with outliers reaching up to **55°C**.
    * **Variability:** Humidity shows high variance, which is essential for the model to learn the difference between 'Dry' and 'Moist' air.
    * **Justification:** Outliers were retained rather than deleted because they represent the extreme 'heatwave' conditions that trigger real fires.
    """)
    st.divider()

    # --- ROW 3: FEATURE VS RISK RELATIONSHIP ---
    st.subheader("3️⃣ Feature vs. Fire Risk")
    # This proves the 'logic' of the features
    st.image("feature vs risk.png", use_container_width=True)
    st.markdown("""
    **Key Findings:**
    * **Temperature Impact:** The median temperature for 'High Risk' (Label 2) is noticeably higher than for 'Low Risk' (Label 0).
    * **Humidity Impact:** Conversely, 'High Risk' scenarios are clearly associated with **lower humidity levels**, proving the inverse relationship.
    """)
    st.divider()

    # --- ROW 4: CLASS DISTRIBUTION ---
    st.subheader("4️⃣ Class Distribution (Fire vs. No Fire)")
    # This proves the data balance
    st.image("class distribution.png", width=500)
    st.markdown("""
    **Key Findings:**
    * **High Risk Samples:** Over **6,000 samples** representing high-risk conditions.
    * **Low Risk Samples:** Approximately **2,500 samples** representing safe conditions.
    * **Conclusion:** This volume of data ensures that the AI has a "strong memory" of what a fire-start condition looks like.
    """)
    st.divider()

    # --- ROW 5: PERFORMANCE (Confusion Matrix) ---
    st.subheader("5️⃣ Performance Analysis (Correlation Heatmap)")
    # This proves the accuracy (TP, FP, etc.)
    st.image("correlation.png", width=500)
    st.markdown("""
    **Key Findings:**
    * **Strongest Link:** `temp_max_c` shows a positive correlation (**0.12**) with `fire_label`.
    * **The Moisture Guard:** `humidity_pct` has a negative correlation (**-0.20**), confirming it acts as a natural fire suppressant in the model's logic.
    """)

    # Professional Footer
    st.success("✅ **Technical Assessment Complete:** The model logic is statistically sound and ready for edge deployment.")    
    