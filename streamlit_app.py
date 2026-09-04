import importlib
import streamlit as st
import pandas as pd

try:
    px = importlib.import_module("plotly.express")
except ImportError:
    px = None

from src.model import (
    load_race_data, build_lap_dataframe, compute_degradation, train_model,
    predict_degradation, forecast_degradation, detect_performance_cliff,
    recommend_pit_window, feature_contributions, _make_input,
)
from src.competitor_engine import (
    build_competitor_snapshot, find_rivals, competitor_strategy_analysis,
    build_position_projection, race_engineer_call,
)
from src.strategy_engine import (
    compare_pit_windows, detect_undercut_overcut, monte_carlo_strategy,
    strategy_recommendation,
)

st.set_page_config(page_title="APEXIA", page_icon="🏎️", layout="wide")
st.title("🏎️ APEXIA")
st.subheader("Intelligence Beyond the Apex")
st.caption("AI-powered Tyre Degradation & Race Strategy Intelligence • Phase 3")
st.divider()


@st.cache_data(show_spinner="Loading FastF1 race data...")
def get_processed_data(year, gp, session_type):
    session = load_race_data(year, gp, session_type)
    return compute_degradation(build_lap_dataframe(session))


@st.cache_resource(show_spinner="Training APEXIA model...")
def get_trained_model(df):
    return train_model(df)


df = get_processed_data(2023, "Bahrain", "R")
model, metrics = get_trained_model(df)

st.sidebar.header("Race Conditions")
compound = st.sidebar.selectbox("Current Tyre", ["SOFT", "MEDIUM", "HARD"], index=1)
new_compound = st.sidebar.selectbox("Pit-stop Tyre", ["SOFT", "MEDIUM", "HARD"], index=2)
tyre_age = st.sidebar.slider("Current Tyre Age", 1, 50, 18)
current_lap = st.sidebar.slider("Current Race Lap", 1, 56, 18)
race_laps = st.sidebar.number_input("Race Distance (laps)", 20, 100, 57)
speed = st.sidebar.slider("Speed at Finish Line", 240, 340, 315)
track_temp = st.sidebar.slider("Track Temperature (°C)", 20, 55, 34)
air_temp = st.sidebar.slider("Air Temperature (°C)", 10, 45, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
rainfall = st.sidebar.checkbox("Rainfall detected", False)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0, 20, 3)
pit_loss = st.sidebar.slider("Pit-lane loss (s)", 15.0, 28.0, 21.5, 0.5)
window_radius = st.sidebar.slider("Pit-window search radius", 2, 6, 4)

# Phase 3: choose the car APEXIA is managing.
drivers = sorted([str(x).upper() for x in df["driver"].dropna().unique() if str(x).strip()])
default_driver = drivers[0] if drivers else "UNKNOWN"
target_driver = st.sidebar.selectbox("APEXIA Car / Driver", drivers, index=0 if default_driver in drivers else 0)

fuel_proxy = max(0.0, min(1.0, 1.0 - current_lap / float(race_laps)))
current_deg = predict_degradation(
    model, tyre_age, compound, speed, track_temp, air_temp, humidity,
    float(rainfall), wind_speed, fuel_proxy
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Tyre Age", f"{tyre_age} laps")
k2.metric("Predicted Loss", f"{current_deg:.2f}s")
k3.metric("Model MAE", f"{metrics['MAE']:.2f}s")
k4.metric("Model R²", f"{metrics['R2']:.2f}")
k5.metric("Fuel Proxy", f"{fuel_proxy:.0%}")

st.divider()
st.header("📈 Tyre Degradation Forecast")
forecast = forecast_degradation(
    model, tyre_age, compound, speed, track_temp, future_laps=min(15, int(race_laps-current_lap)),
    air_temp=air_temp, humidity=humidity, rainfall=float(rainfall),
    wind_speed=wind_speed, current_race_lap=current_lap
)

if px and not forecast.empty:
    fig = px.line(forecast, x="lap", y="predicted_degradation", markers=True,
                  title="Predicted tyre-attributed performance loss")
    fig.update_layout(xaxis_title="Race Lap", yaxis_title="Performance Loss (s)")
    st.plotly_chart(fig, use_container_width=True)

cliff = detect_performance_cliff(forecast)
if cliff:
    st.error(f"⚠️ PERFORMANCE CLIFF: predicted around Lap {cliff['lap']} ({cliff['severity']} acceleration).")
else:
    st.success("🟢 No sharp performance cliff detected in the forecast window.")

pit_window = recommend_pit_window(forecast, cliff)
if pit_window:
    st.info(f"🏁 Initial degradation-based pit window: **Lap {pit_window['start']}–{pit_window['end']}** — {pit_window['reason']}")

st.divider()
st.header("🏁 Phase 2 — Race Strategy Engine")
st.write("APEXIA now evaluates pit timing by projected race time, not tyre degradation alone.")

candidate_start = max(current_lap + 1, (pit_window["start"] if pit_window else current_lap + 2) - window_radius)
candidate_end = min(int(race_laps) - 1, (pit_window["end"] if pit_window else current_lap + 6) + window_radius)
candidate_laps = range(candidate_start, candidate_end + 1)

window_df = compare_pit_windows(
    model, df, current_lap, tyre_age, compound, speed, track_temp, int(race_laps),
    candidate_laps, new_compound=new_compound, air_temp=air_temp,
    humidity=humidity, rainfall=float(rainfall), wind_speed=wind_speed,
    pit_loss=pit_loss, strategy="BALANCED"
)

if window_df.empty:
    st.warning("Not enough remaining race distance to evaluate a pit window.")
else:
    mc_df = monte_carlo_strategy(window_df, simulations=500, uncertainty_seconds=0.35)
    recommendation = strategy_recommendation(window_df, mc_df)
    signal = detect_undercut_overcut(window_df, current_lap)

    a, b, c, d = st.columns(4)
    a.metric("Recommended Pit", f"Lap {recommendation['pit_lap']}")
    b.metric("Confidence", f"{recommendation['confidence']:.0%}")
    c.metric("Best Projected Delta", f"{window_df.iloc[0]['delta_to_best']:.2f}s")
    d.metric("Strategy Signal", signal["signal"])

    st.success(f"🧠 **APEXIA:** {recommendation['message']}")
    st.info(f"**{signal['signal']}** — {signal['message']}")

    left, right = st.columns(2)
    with left:
        st.subheader("Pit-window optimisation")
        if px:
            fig = px.line(window_df, x="pit_lap", y="delta_to_best", markers=True,
                          title="Projected penalty vs best pit lap")
            fig.update_layout(xaxis_title="Pit Lap", yaxis_title="Delta to Best (s)")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(window_df.round(3), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Monte Carlo confidence")
        if px:
            fig = px.bar(mc_df, x="pit_lap", y="win_probability", text_auto=".0%",
                         title="Probability each pit lap is optimal")
            fig.update_layout(xaxis_title="Pit Lap", yaxis_title="Win Probability")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(mc_df.round(3), use_container_width=True, hide_index=True)

st.divider()
st.header("🏎️ Phase 3 — Competitor-Aware Race Intelligence")
st.write("APEXIA now evaluates your car against the nearest cars ahead and behind using the available race timing data.")

snapshot = build_competitor_snapshot(df, int(current_lap), target_driver)
if snapshot.empty:
    st.warning("No competitor timing data is available at the selected race lap.")
else:
    rivals = find_rivals(snapshot, target_driver, max_gap=10.0)
    tactical = competitor_strategy_analysis(
        model, df, int(current_lap), int(race_laps), target_driver,
        target_new_compound=new_compound, pit_loss=pit_loss, horizon=6,
    )
    selected_pit = int(recommendation["pit_lap"]) if not window_df.empty else min(int(race_laps)-1, int(current_lap)+3)
    projection = build_position_projection(
        model, df, int(current_lap), int(race_laps), target_driver,
        selected_pit, new_compound, pit_loss=pit_loss, horizon=5,
    )
    engineer = race_engineer_call(snapshot, tactical, projection, target_driver)

    ea, eb, ec, ed = st.columns(4)
    target_snap = snapshot[snapshot["driver"] == str(target_driver).upper()]
    est_pos = int(target_snap.iloc[0]["relative_position_est"]) if not target_snap.empty else None
    ahead_gap = rivals["ahead"]["gap"] if rivals.get("ahead") else None
    behind_gap = rivals["behind"]["gap"] if rivals.get("behind") else None
    ea.metric("Estimated Position", f"P{est_pos}" if est_pos else "—")
    eb.metric("Car Ahead Gap", f"{ahead_gap:.1f}s" if ahead_gap is not None else "—")
    ec.metric("Car Behind Gap", f"{behind_gap:.1f}s" if behind_gap is not None else "—")
    ed.metric("Selected Pit", f"Lap {selected_pit}")

    if engineer["severity"] == "HIGH":
        st.error(f"🚨 **{engineer['signal']}** — {engineer['message']}")
    elif engineer["severity"] == "MEDIUM":
        st.warning(f"⚠️ **{engineer['signal']}** — {engineer['message']}")
    else:
        st.success(f"🧠 **RACE ENGINEER:** {engineer['message']}")

    left, right = st.columns(2)
    with left:
        st.subheader("Nearest rivals")
        rival_rows = []
        for relation in ["ahead", "behind"]:
            item = rivals.get(relation)
            if item:
                rival_rows.append({
                    "relation": relation.upper(),
                    "driver": item["driver"],
                    "gap_s": item["gap"],
                    "tyre": item["compound"],
                    "tyre_age": item["tyre_age"],
                    "position": item["position"],
                })
        if rival_rows:
            st.dataframe(pd.DataFrame(rival_rows).round(2), use_container_width=True, hide_index=True)
        else:
            st.info("No rival within the 10-second tactical window.")

    with right:
        st.subheader("Tactical opportunities")
        if tactical.empty:
            st.info("No tactical rival scenarios could be evaluated.")
        else:
            st.dataframe(tactical.round(2), use_container_width=True, hide_index=True)

    st.subheader("Projected race order after selected pit")
    if projection.empty:
        st.info("Position projection unavailable.")
    else:
        st.dataframe(projection.round(2), use_container_width=True, hide_index=True)
        if px:
            plot_df = projection.head(10).copy()
            plot_df["driver_label"] = plot_df["driver"].astype(str)
            fig = px.bar(plot_df, x="driver_label", y="projected_time", text="projected_position",
                         title=f"Estimated order {min(5, int(race_laps-current_lap))}-lap horizon after Lap {selected_pit} pit")
            fig.update_layout(xaxis_title="Driver", yaxis_title="Projected Time (s)")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Phase 3 assumptions"):
        st.write("Gap is estimated from cumulative recorded lap times when official timing gaps are not directly available in the processed dataframe.")
        st.write("Competitor projections use the same degradation model and a short tactical horizon; they are not live GPS/telemetry predictions.")
        st.write("Use this layer as a strategy-support signal, not as an official classification feed.")

st.divider()
st.header("🎯 What-If Driving Strategy")
strategies = ["PUSH", "BALANCED", "MANAGE"]
strategy_rows = []
for strategy in strategies:
    sf = forecast_degradation(
        model, tyre_age, compound, speed, track_temp,
        future_laps=min(10, int(race_laps-current_lap)), air_temp=air_temp,
        humidity=humidity, rainfall=float(rainfall), wind_speed=wind_speed,
        current_race_lap=current_lap, strategy=strategy
    )
    if not sf.empty:
        strategy_rows.append(sf.assign(strategy=strategy))

if strategy_rows:
    all_strategy = pd.concat(strategy_rows, ignore_index=True)
    finals = all_strategy.groupby("strategy")["predicted_degradation"].last().sort_values()
    c1, c2, c3 = st.columns(3)
    for col, name in zip([c1, c2, c3], strategies):
        col.metric(name, f"{finals.get(name, 0):.2f}s")
    if px:
        fig = px.line(all_strategy, x="lap", y="predicted_degradation", color="strategy", markers=True,
                      title="Tyre degradation under driving-style scenarios")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.header("🧠 Explainability")
input_row = _make_input(tyre_age, compound, speed, track_temp, air_temp, humidity,
                        float(rainfall), wind_speed, fuel_proxy)
contrib = feature_contributions(model, input_row)
display_names = {
    "tyre_age":"Tyre Age", "compound_encoded":"Compound", "speed":"Speed",
    "track_temp":"Track Temperature", "air_temp":"Air Temperature", "humidity":"Humidity",
    "rainfall":"Rainfall", "wind_speed":"Wind Speed", "fuel_proxy":"Fuel Proxy",
    "is_safety_car":"Safety Car", "is_vsc":"VSC",
}
contrib["Feature"] = contrib["Feature"].map(display_names).fillna(contrib["Feature"])
if px:
    fig = px.bar(contrib, x="Contribution", y="Feature", orientation="h",
                 title="Local model contribution (perturbation-based)")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 Data & model diagnostics"):
    st.write(f"Rows: {len(df):,}")
    st.write(f"Drivers: {df['driver'].nunique()}")
    st.write(f"Stints: {df['stint_id'].nunique()}")
    st.write(f"Training rows: {metrics['train_rows']:,}")
    st.write(f"Test rows: {metrics['test_rows']:,}")
    cols = ["driver", "lap", "compound", "tyre_age", "lap_time", "degradation",
            "degradation_rate", "track_temp", "air_temp", "humidity", "rainfall",
            "is_safety_car", "is_vsc"]
    st.dataframe(df[[c for c in cols if c in df.columns]].tail(30), use_container_width=True)

st.caption(
    "Phase 2 strategy outputs are simulation estimates. Pit loss, compound pace offsets and fuel effect "
    "are configurable assumptions; true fuel mass and live competitor telemetry are not available in the current FastF1 input."
)
