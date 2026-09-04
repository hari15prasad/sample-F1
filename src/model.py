import os
import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingImports]
import fastf1  # type: ignore[reportMissingImports]

from sklearn.ensemble import RandomForestRegressor  # type: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split  # type: ignore[reportMissingImports]
from sklearn.metrics import mean_absolute_error, r2_score  # type: ignore[reportMissingImports]

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


# --------------------------------------------------
# 1. Load real race telemetry
# --------------------------------------------------

def load_race_data(year=2023, gp="Bahrain", session_type="R"):
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def build_lap_dataframe(session):
    laps = session.laps.copy()
    weather = session.weather_data.copy()

    laps["lap_time"] = laps["LapTime"].dt.total_seconds()
    laps = laps.dropna(subset=["lap_time", "TyreLife", "Compound"])
    laps = laps[laps["lap_time"] < laps["lap_time"].quantile(0.98)]

    laps = pd.merge_asof(
        laps.sort_values("Time"),
        weather.sort_values("Time")[["Time", "TrackTemp"]],
        on="Time",
        direction="nearest"
    )

    laps["SpeedFL"] = laps["SpeedFL"].fillna(laps["SpeedFL"].median())

    df = laps.rename(columns={
        "LapNumber": "lap",
        "TyreLife": "tyre_age",
        "Compound": "compound",
        "SpeedFL": "speed",
        "TrackTemp": "track_temp"
    })[["lap", "tyre_age", "compound", "speed", "track_temp", "Stint", "lap_time"]]

    # Force plain DataFrame — strips fastf1's Laps subclass so it's
    # hashable for Streamlit caching and behaves like normal pandas
    return pd.DataFrame(df).reset_index(drop=True)


def compute_degradation(df):
    df = df.copy()
    df["degradation"] = df.groupby("Stint")["lap_time"].transform(lambda s: s - s.min())
    return pd.DataFrame(df)


# --------------------------------------------------
# 2. Prepare ML features
# --------------------------------------------------

def prepare_features(df):
    df = df.copy()

    df["compound_encoded"] = df["compound"].map({
        "SOFT": 3,
        "MEDIUM": 2,
        "HARD": 1
    })
    df = df.dropna(subset=["compound_encoded"])

    features = ["tyre_age", "compound_encoded", "speed", "track_temp"]

    X = df[features]
    y = df["degradation"]

    return X, y


# --------------------------------------------------
# 3. Train model
# --------------------------------------------------

def train_model(df):
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "MAE": round(mean_absolute_error(y_test, predictions), 3),
        "R2": round(r2_score(y_test, predictions), 3)
    }

    return model, metrics


# --------------------------------------------------
# 4. Predict degradation
# --------------------------------------------------

def predict_degradation(model, tyre_age, compound, speed, track_temp):
    compound_encoded = {"SOFT": 3, "MEDIUM": 2, "HARD": 1}[compound.upper()]

    data = pd.DataFrame([{
        "tyre_age": tyre_age,
        "compound_encoded": compound_encoded,
        "speed": speed,
        "track_temp": track_temp
    }])

    prediction = model.predict(data)[0]
    return round(float(prediction), 3)


# --------------------------------------------------
# 5. Forecast future degradation
# --------------------------------------------------

def forecast_degradation(model, current_lap, compound, speed, track_temp, future_laps=15):
    results = []

    for i in range(1, future_laps + 1):
        future_age = current_lap + i
        degradation = predict_degradation(model, future_age, compound, speed, track_temp)
        results.append({"lap": future_age, "predicted_degradation": degradation})

    return pd.DataFrame(results)


# --------------------------------------------------
# 6. What-if strategy simulator
# --------------------------------------------------

def simulate_strategy(model, current_lap, compound, speed, track_temp, strategy):
    adjustments = {
        "PUSH": {"speed": 4},
        "BALANCED": {"speed": 0},
        "MANAGE": {"speed": -4}
    }

    a = adjustments[strategy]

    forecast = forecast_degradation(
        model,
        current_lap,
        compound,
        speed + a["speed"],
        track_temp,
        future_laps=10
    )

    final_degradation = forecast["predicted_degradation"].iloc[-1]
    return forecast, round(float(final_degradation), 3)