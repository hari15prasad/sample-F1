import importlib
import streamlit as st  # type: ignore[import-not-found]

try:
    px = importlib.import_module("plotly.express")
except ImportError:  # pragma: no cover
    px = None

from src.model import (
    load_race_data,
    build_lap_dataframe,
    compute_degradation,
    train_model,
    predict_degradation,
    forecast_degradation,
    simulate_strategy
)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="APEXIA",
    page_icon="🏎️",
    layout="wide"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏎️ APEXIA")
st.subheader("Intelligence Beyond the Apex")

st.markdown(
    "### AI-powered Tyre Degradation & Race Strategy Intelligence"
)

st.divider()


# --------------------------------------------------
# LOAD DATA + MODEL (cached)
# --------------------------------------------------

@st.cache_data
def get_processed_data(year, gp, session_type):
    session = load_race_data(year, gp, session_type)
    df = build_lap_dataframe(session)
    df = compute_degradation(df)
    return df

@st.cache_resource
def get_trained_model(df):
    return train_model(df)

df = get_processed_data(2023, "Bahrain", "R")
model, metrics = get_trained_model(df)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Race Conditions")

compound = st.sidebar.selectbox(
    "Tyre Compound",
    ["SOFT", "MEDIUM", "HARD"]
)

tyre_age = st.sidebar.slider(
    "Current Tyre Age",
    1,
    50,
    18
)

speed = st.sidebar.slider(
    "Speed at Finish Line",
    280,
    330,
    315
)

track_temp = st.sidebar.slider(
    "Track Temperature",
    20,
    50,
    34
)


# --------------------------------------------------
# CURRENT TYRE STATE
# --------------------------------------------------

current_deg = predict_degradation(
    model,
    tyre_age,
    compound,
    speed,
    track_temp
)


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Tyre Age", f"{tyre_age} laps")
col2.metric("Degradation", f"{current_deg:.2f}s")
col3.metric("Model MAE", metrics["MAE"])
col4.metric("Model R²", metrics["R2"])

st.divider()


# --------------------------------------------------
# FORECAST
# --------------------------------------------------

st.header("📈 Tyre Degradation Forecast")

forecast = forecast_degradation(
    model,
    tyre_age,
    compound,
    speed,
    track_temp,
    future_laps=15
)

fig = px.line(
    forecast,
    x="lap",
    y="predicted_degradation",
    markers=True,
    title="Predicted degradation over upcoming laps"
)

fig.update_layout(
    xaxis_title="Future Lap",
    yaxis_title="Predicted Performance Loss (s)"
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# WHAT-IF STRATEGY
# --------------------------------------------------

st.header("🎯 What-If Race Strategy")

st.write("How does pace style affect predicted degradation?")

strategies = ["PUSH", "BALANCED", "MANAGE"]
strategy_results = {}

for strategy in strategies:
    _, final_deg = simulate_strategy(
        model,
        tyre_age,
        compound,
        speed,
        track_temp,
        strategy
    )
    strategy_results[strategy] = final_deg

c1, c2, c3 = st.columns(3)
c1.metric("🔥 PUSH", strategy_results["PUSH"])
c2.metric("⚖️ BALANCED", strategy_results["BALANCED"])
c3.metric("🛞 MANAGE", strategy_results["MANAGE"])


# --------------------------------------------------
# RECOMMENDATION
# --------------------------------------------------

best_strategy = min(strategy_results, key=strategy_results.get)

st.success(
    f"APEXIA recommendation: **{best_strategy}** "
    f"currently produces the lowest predicted degradation "
    f"over the simulated window."
)


# --------------------------------------------------
# EXPLANATION
# --------------------------------------------------

st.header("🧠 Why is degradation increasing?")

importance = model.feature_importances_

features = ["Tyre Age", "Compound", "Speed", "Track Temperature"]

importance_df = {
    "Feature": features,
    "Importance": importance
}

importance_fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Model feature importance"
)

st.plotly_chart(importance_fig, use_container_width=True)


# --------------------------------------------------
# PIT WINDOW
# --------------------------------------------------

st.header("🏁 Strategy Insight")

if current_deg < 1.0:
    message = "Tyre performance is currently healthy. Continuing the stint may be viable."
elif current_deg < 2.0:
    message = "Moderate degradation detected. Monitor tyre performance closely."
else:
    message = "High degradation detected. A pit-window decision should be considered."

st.info(message)

st.caption(
    "APEXIA is a decision-support prototype. "
    "Data shown is real lap and telemetry data from the 2023 Bahrain Grand Prix, "
    "sourced via FastF1."
)