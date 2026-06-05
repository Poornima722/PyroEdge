# 🌲 PyroEdge: Edge-AI Enabled Forest Fire Risk Prediction Framework

PyroEdge is an automated, real-time Internet of Things (IoT) prototype designed to assess micro-climatic wildfire risks. By decoupling distributed edge sensor telemetry from centralized heavy computation, PyroEdge captures local atmospheric variations and performs live Machine Learning inference directly on a centralized localized dashboard.

---

## 🛠️ System Architecture & Pipeline

1. **Hardware Telemetry Layer:** Distributed **ESP8266 Microcontroller Nodes** continuously capture environmental metrics via dedicated sensor modules (Temperature, Humidity, and Rainfall).
2. **Network Protocol Layer:** The edge nodes transmit real-time telemetry packets over local network sockets to a central processing hub via the **HTTP Protocol** as structured JSON streams.
3. **Edge Inference Hub:** A **Raspberry Pi** acts as the localized server, parsing the inbound data payload. It runs a pre-trained **Random Forest Classifier** to compute dynamic hazard evaluation flags.
4. **Interactive UI Layer:** A dual-tabbed **Streamlit Dashboard** visualizes the localized data pipelines, live alert logging systems, and deep machine learning validation metrics.

---

## 📊 Dashboard Implementation Framework

The PyroEdge system frontend features a robust, responsive workspace split into two distinct operational perspectives:

### 1. 🔥 Tab 01: Live Monitoring & Inference Dashboard
* **Dynamic Node Matrices:** Separated real-time data cards updating every 3 seconds for **Node 01** and **Node 02**, highlighting Temperature (°C), Humidity (%), Rainfall state, and an AI-computed **Fire Risk Status Flag**.
* **Temporal Series Graphics:** Interactive Plotly lines mapping continuous time-series temperature trends for individual deployment zones.
* **Hybrid Guardrail Execution:** Employs a physical constraint system matching statistical inference with real-world thresholds to guarantee zero false-alarm indexing during downpours or cool anomalies.
* **Persistent Incident History Log:** A localized session-state memory queue tracking runtime alert triggers, system state transitions, and resolution timestamps.

### 2. 📊 Tab 02: Explanatory Data Analytics (EDA) & Justification
* **Feature Correlation Analysis:** A custom heatmap indexing the mathematical interactions between ambient heat, dry-moisture indices, and rainfall occurrences.
* **Target Distribution Matrix:** Visual bar arrays plotting the density of historical target labels ($0 = \text{Safe}$, $1 = \text{Actionable Risk}$) used to evaluate feature-space composition.
* **Outlier Profiling:** Box-and-whisker distributions charting continuous variables to map data extremes.
* **System Evaluation Graphics:** A $2 \times 2$ Confusion Matrix and exhaustive classification reports verifying the model's optimized **79.87% evaluation accuracy**.

---

## 📁 Repository Structure

```text
├── final_dataset.csv       # Preprocessed historical weather snapshot data
├── model.ipynb             # Interactive Jupyter Notebook for ML training & tuning
├── forest_fire_model.pkl   # Serialized, high-capacity binary Random Forest model
├── app.py                  # Main Streamlit Dashboard interactive script
└── README.md               # Framework technical documentation
