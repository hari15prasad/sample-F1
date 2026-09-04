import os
from typing import Dict, Tuple

import fastf1
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

COMPOUND_ENCODING = {"SOFT": 3, "MEDIUM": 2, "HARD": 1, "INTERMEDIATE": 0, "WET": -1}
STRATEGY_SPEED_DELTA = {"PUSH": 4, "BALANCED": 0, "MANAGE": -4}


def load_race_data(year=2023, gp="Bahrain", session_type="R"):
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def _track_status_flags(track_status: pd.Series) -> pd.DataFrame:
    status = track_status.fillna("1").astype(str)
    return pd.DataFrame({
        "is_green": status.str.contains(r"(^|,)1(,|$)", regex=True),
        "is_yellow": status.str.contains(r"(^|,)2(,|$)", regex=True),
        "is_safety_car": status.str.contains(r"(^|,)4(,|$)", regex=True),
        "is_red_flag": status.str.contains(r"(^|,)5(,|$)", regex=True),
        "is_vsc": status.str.contains(r"(^|,)6(,|$)", regex=True),
    }, index=track_status.index).astype(int)


def build_lap_dataframe(session):
    laps = pd.DataFrame(session.laps.copy())
    weather = pd.DataFrame(session.weather_data.copy())

    laps["lap_time"] = laps["LapTime"].dt.total_seconds()
    laps = laps.dropna(subset=["lap_time", "TyreLife", "Compound", "LapNumber"])

    # Remove obvious outliers/pit-lap distortions.
    laps = laps[laps["lap_time"] < laps["lap_time"].quantile(0.98)].copy()

    weather_cols = [
        c for c in [
            "Time", "TrackTemp", "AirTemp", "Humidity", "Rainfall",
            "WindSpeed", "WindDirection"
        ] if c in weather.columns
    ]
    if "Time" in weather_cols and "Time" in laps.columns:
        laps = laps.sort_values("Time")
        weather = weather.sort_values("Time")
        laps = pd.merge_asof(
            laps,
            weather[weather_cols],
            on="Time",
            direction="nearest",
        )

    wanted = {
        "LapNumber": "lap",
        "TyreLife": "tyre_age",
        "Compound": "compound",
        "SpeedFL": "speed",
        "TrackTemp": "track_temp",
        "AirTemp": "air_temp",
        "Humidity": "humidity",
        "Rainfall": "rainfall",
        "WindSpeed": "wind_speed",
        "WindDirection": "wind_direction",
        "Stint": "stint",
        "Driver": "driver",
        "Team": "team",
        "Position": "position",
        "TrackStatus": "track_status",
        "Sector1Time": "sector1_time",
        "Sector2Time": "sector2_time",
        "Sector3Time": "sector3_time",
    }
    available = {k: v for k, v in wanted.items() if k in laps.columns}
    df = laps.rename(columns=available)

    defaults = {
        "driver": "UNKNOWN",
        "team": "UNKNOWN",
        "position": np.nan,
        "stint": 0,
        "track_status": "1",
        "speed": np.nan,
        "track_temp": np.nan,
        "air_temp": np.nan,
        "humidity": np.nan,
        "rainfall": 0.0,
        "wind_speed": np.nan,
        "wind_direction": np.nan,
        "sector1_time": np.nan,
        "sector2_time": np.nan,
        "sector3_time": np.nan,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["driver"] = df["driver"].fillna("UNKNOWN")
    df["compound"] = df["compound"].astype(str).str.upper()
    df["compound_encoded"] = df["compound"].map(COMPOUND_ENCODING)
    df["stint_id"] = df["driver"].astype(str) + "_" + df["stint"].astype(str)

    # A simple race fuel proxy. FastF1 timing data does not expose true fuel mass,
    # so this is deliberately labelled as a proxy rather than ground truth.
    df["fuel_proxy"] = 1.0 - (
        df["lap"] / max(float(df["lap"].max()), 1.0)
    )

    flags = _track_status_flags(df["track_status"])
    for col in flags.columns:
        df[col] = flags[col]

    numeric_cols = [
        "speed", "track_temp", "air_temp", "humidity", "rainfall",
        "wind_speed", "wind_direction", "sector1_time", "sector2_time",
        "sector3_time", "fuel_proxy"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Do not train on non-racing laps, but retain the flags for dashboard use.
    df = df[df["is_red_flag"] == 0].copy()
    return pd.DataFrame(df).reset_index(drop=True)


def compute_degradation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def add_stint_metrics(group):
        group = group.sort_values("lap").copy()
        # Use the best representative early-stint pace rather than the first lap.
        baseline = group["lap_time"].rolling(3, min_periods=1).min().expanding().min()
        group["baseline_lap_time"] = baseline
        group["degradation"] = (group["lap_time"] - baseline).clip(lower=0)
        group["lap_delta"] = group["lap_time"].diff()
        return group

    df = (
        df.groupby("stint_id", group_keys=False)
        .apply(add_stint_metrics)
        .reset_index(drop=True)
    )

    # Sector-level degradation relative to the best sector in each stint.
    for sector in ["sector1_time", "sector2_time", "sector3_time"]:
        if sector in df:
            best = df.groupby("stint_id")[sector].transform("min")
            df[f"{sector}_degradation"] = (df[sector] - best).clip(lower=0)

    # Smoothed degradation rate (seconds/lap).
    df["degradation_rate"] = (
        df.groupby("stint_id")["degradation"]
        .transform(lambda s: s.diff().rolling(3, min_periods=1).mean())
        .fillna(0)
    )
    return pd.DataFrame(df)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()
    df = df.dropna(subset=["degradation", "compound_encoded", "tyre_age"])

    features = [
        "tyre_age", "compound_encoded", "speed", "track_temp",
        "air_temp", "humidity", "rainfall", "wind_speed",
        "fuel_proxy", "is_safety_car", "is_vsc"
    ]
    for feature in features:
        if feature not in df:
            df[feature] = 0
        df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0)

    X = df[features]
    y = df["degradation"].astype(float)
    groups = df["stint_id"]
    return X, y, groups


def train_model(df: pd.DataFrame):
    X, y, groups = prepare_features(df)

    if len(X) < 20 or groups.nunique() < 2:
        raise ValueError("Not enough clean stint data to train APEXIA.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    predictions = model.predict(X.iloc[test_idx])
    metrics = {
        "MAE": round(float(mean_absolute_error(y.iloc[test_idx], predictions)), 3),
        "R2": round(float(r2_score(y.iloc[test_idx], predictions)), 3),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
    }
    return model, metrics


def _make_input(
    tyre_age,
    compound,
    speed,
    track_temp,
    air_temp=25.0,
    humidity=50.0,
    rainfall=0.0,
    wind_speed=0.0,
    fuel_proxy=0.5,
    is_safety_car=0,
    is_vsc=0,
):
    return pd.DataFrame([{
        "tyre_age": tyre_age,
        "compound_encoded": COMPOUND_ENCODING.get(str(compound).upper(), 1),
        "speed": speed,
        "track_temp": track_temp,
        "air_temp": air_temp,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "fuel_proxy": fuel_proxy,
        "is_safety_car": is_safety_car,
        "is_vsc": is_vsc,
    }])


def predict_degradation(model, tyre_age, compound, speed, track_temp,
                         air_temp=25.0, humidity=50.0, rainfall=0.0,
                         wind_speed=0.0, fuel_proxy=0.5,
                         is_safety_car=0, is_vsc=0):
    data = _make_input(
        tyre_age, compound, speed, track_temp, air_temp, humidity,
        rainfall, wind_speed, fuel_proxy, is_safety_car, is_vsc
    )
    return round(float(max(0.0, model.predict(data)[0])), 3)


def forecast_degradation(model, current_tyre_age, compound, speed, track_temp,
                         future_laps=15, air_temp=25.0, humidity=50.0,
                         rainfall=0.0, wind_speed=0.0, current_race_lap=None,
                         strategy="BALANCED"):
    results = []
    speed_delta = STRATEGY_SPEED_DELTA.get(strategy.upper(), 0)
    race_lap = current_race_lap if current_race_lap is not None else current_tyre_age

    for i in range(1, future_laps + 1):
        future_age = current_tyre_age + i
        fuel_proxy = max(0.0, 1.0 - (race_lap + i) / max(race_lap + future_laps + 10, 1))
        degradation = predict_degradation(
            model, future_age, compound, speed + speed_delta, track_temp,
            air_temp, humidity, rainfall, wind_speed, fuel_proxy
        )
        results.append({
            "lap": race_lap + i,
            "tyre_age": future_age,
            "predicted_degradation": degradation,
            "strategy": strategy.upper(),
        })

    return pd.DataFrame(results)


def detect_performance_cliff(forecast: pd.DataFrame, threshold=0.18):
    if forecast.empty:
        return None

    values = forecast["predicted_degradation"].to_numpy()
    if len(values) < 3:
        return None

    slope = np.gradient(values)
    for i, value in enumerate(slope):
        if i >= 1 and value >= threshold:
            return {
                "lap": int(forecast.iloc[i]["lap"]),
                "rate": round(float(value), 3),
                "severity": "HIGH" if value >= threshold * 1.75 else "MEDIUM",
            }
    return None


def recommend_pit_window(forecast: pd.DataFrame, cliff=None):
    if forecast.empty:
        return None

    if cliff:
        cliff_lap = cliff["lap"]
        return {
            "start": max(int(forecast["lap"].min()), cliff_lap - 2),
            "end": max(int(forecast["lap"].min()), cliff_lap),
            "reason": "Pit before or at the predicted tyre-performance cliff."
        }

    # If no cliff is detected, use the first lap where degradation exceeds 2s.
    high = forecast[forecast["predicted_degradation"] >= 2.0]
    if not high.empty:
        lap = int(high.iloc[0]["lap"])
        return {"start": max(int(forecast["lap"].min()), lap - 1),
                "end": lap + 1,
                "reason": "Degradation has entered a high-loss region."}

    return {
        "start": int(forecast["lap"].max()) - 2,
        "end": int(forecast["lap"].max()),
        "reason": "No severe degradation detected; extend the stint and monitor."
    }


def simulate_strategy(model, current_lap, tyre_age, compound, speed, track_temp,
                      strategy, air_temp=25.0, humidity=50.0, rainfall=0.0,
                      wind_speed=0.0, future_laps=10):
    forecast = forecast_degradation(
        model,
        current_tyre_age=tyre_age,
        compound=compound,
        speed=speed,
        track_temp=track_temp,
        future_laps=future_laps,
        air_temp=air_temp,
        humidity=humidity,
        rainfall=rainfall,
        wind_speed=wind_speed,
        current_race_lap=current_lap,
        strategy=strategy,
    )
    final_deg = float(forecast["predicted_degradation"].iloc[-1])
    return forecast, round(final_deg, 3)


def feature_contributions(model, inputs: pd.DataFrame) -> pd.DataFrame:
    """Model-agnostic local feature contribution approximation.

    This intentionally avoids making SHAP a hard dependency. If SHAP is installed,
    the UI can replace this with TreeExplainer values.
    """
    baseline = float(model.predict(inputs)[0])
    rows = []
    for feature in inputs.columns:
        perturbed = inputs.copy()
        value = float(perturbed.iloc[0][feature])
        delta = max(abs(value) * 0.10, 0.1)
        perturbed.iloc[0, perturbed.columns.get_loc(feature)] = value + delta
        changed = float(model.predict(perturbed)[0])
        rows.append({
            "Feature": feature,
            "Contribution": round(changed - baseline, 4),
        })
    return pd.DataFrame(rows).sort_values("Contribution", key=abs, ascending=False)



def forecast_strategy_profiles(model, current_lap, tyre_age, compound, speed, track_temp,
                               future_laps=10, air_temp=25.0, humidity=50.0,
                               rainfall=0.0, wind_speed=0.0):
    """Return PUSH/BALANCED/MANAGE forecasts in one dataframe."""
    frames = []
    for strategy in ["PUSH", "BALANCED", "MANAGE"]:
        frame = forecast_degradation(
            model, tyre_age, compound, speed, track_temp, future_laps,
            air_temp, humidity, rainfall, wind_speed, current_lap, strategy
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
