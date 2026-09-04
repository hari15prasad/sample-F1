"""APEXIA Phase 2 race-strategy engine.

The engine converts tyre degradation forecasts into approximate race-time
outcomes. It is intentionally explicit about assumptions: pit loss, tyre
compound pace offsets and fuel progression are configurable estimates, not
telemetry ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .model import COMPOUND_ENCODING, forecast_degradation


COMPOUND_PACE_OFFSET = {
    # Approximate dry-tyre pace offsets in seconds relative to the chosen
    # reference compound. These are defaults and should be calibrated per track.
    "SOFT": -0.55,
    "MEDIUM": 0.00,
    "HARD": 0.45,
    "INTERMEDIATE": 3.0,
    "WET": 6.0,
}

DEFAULT_PIT_LOSS = 21.5
DEFAULT_SC_PIT_LOSS = 10.0
DEFAULT_VSC_PIT_LOSS = 14.0


@dataclass(frozen=True)
class StrategyConfig:
    race_laps: int
    pit_lap: Optional[int] = None
    new_compound: Optional[str] = None
    pit_loss: float = DEFAULT_PIT_LOSS
    sc_pit_loss: float = DEFAULT_SC_PIT_LOSS
    vsc_pit_loss: float = DEFAULT_VSC_PIT_LOSS
    tyre_reset_age: int = 1


@dataclass
class StrategyResult:
    name: str
    total_time_loss: float
    pit_lap: Optional[int]
    compound_after_pit: Optional[str]
    projected_finish_delta: float
    cliff_lap: Optional[int]
    pit_window: Optional[tuple[int, int]]
    confidence: float
    rationale: str


def estimate_baseline_pace(df: pd.DataFrame, compound: str, driver: str | None = None) -> float:
    """Estimate representative green-flag lap pace for a compound.

    Uses the fastest 30% of available laps after filtering obvious outliers.
    Driver-specific data is preferred when enough observations exist.
    """
    work = df.copy()
    work["compound"] = work["compound"].astype(str).str.upper()
    subset = work[work["compound"] == compound.upper()].copy()
    if driver is not None and "driver" in work:
        driver_subset = subset[subset["driver"].astype(str) == str(driver)]
        if len(driver_subset) >= 5:
            subset = driver_subset

    if "is_green" in subset:
        green = subset[subset["is_green"] == 1]
        if len(green) >= 5:
            subset = green

    if subset.empty:
        # Bahrain 2023 dry-race pace fallback. This is only a simulation
        # baseline, never presented as measured telemetry.
        fallback = {"SOFT": 92.0, "MEDIUM": 93.0, "HARD": 94.0}
        return fallback.get(compound.upper(), 93.0)

    times = pd.to_numeric(subset["lap_time"], errors="coerce").dropna()
    times = times[(times > times.quantile(0.02)) & (times < times.quantile(0.90))]
    if times.empty:
        return float(subset["lap_time"].median())

    n = max(3, int(np.ceil(len(times) * 0.30)))
    return float(times.nsmallest(n).median())


def _track_status_for_lap(df: pd.DataFrame, lap: int) -> tuple[bool, bool]:
    row = df[df["lap"].round().astype(int) == int(lap)]
    if row.empty:
        return False, False
    return bool(row["is_safety_car"].max()), bool(row["is_vsc"].max())


def _fuel_proxy_for_lap(lap: int, race_laps: int) -> float:
    return max(0.0, min(1.0, 1.0 - float(lap) / max(float(race_laps), 1.0)))


def simulate_stint(
    model,
    current_lap: int,
    current_tyre_age: int,
    compound: str,
    speed: float,
    track_temp: float,
    end_lap: int,
    race_laps: int,
    air_temp: float = 25.0,
    humidity: float = 50.0,
    rainfall: float = 0.0,
    wind_speed: float = 0.0,
    pace_reference: float = 93.0,
    compound_reference: str = "MEDIUM",
    strategy: str = "BALANCED",
) -> pd.DataFrame:
    """Project lap time for each lap in a stint."""
    if end_lap < current_lap + 1:
        return pd.DataFrame(columns=["lap", "tyre_age", "predicted_degradation", "projected_lap_time"])

    horizon = end_lap - current_lap
    forecast = forecast_degradation(
        model,
        current_tyre_age=current_tyre_age,
        compound=compound,
        speed=speed,
        track_temp=track_temp,
        future_laps=horizon,
        air_temp=air_temp,
        humidity=humidity,
        rainfall=rainfall,
        wind_speed=wind_speed,
        current_race_lap=current_lap,
        strategy=strategy,
    )
    compound = compound.upper()
    offset = COMPOUND_PACE_OFFSET.get(compound, 0.0) - COMPOUND_PACE_OFFSET.get(compound_reference.upper(), 0.0)
    # Fuel proxy is a progress indicator. We only apply a small, configurable
    # pace correction so the simulation does not pretend to know fuel mass.
    rows = []
    for _, r in forecast.iterrows():
        lap = int(r["lap"])
        fuel_proxy = _fuel_proxy_for_lap(lap, race_laps)
        fuel_effect = 0.20 * (1.0 - fuel_proxy)
        rows.append({
            "lap": lap,
            "tyre_age": int(r["tyre_age"]),
            "predicted_degradation": float(r["predicted_degradation"]),
            "projected_lap_time": float(pace_reference + offset + fuel_effect + r["predicted_degradation"]),
            "fuel_proxy": fuel_proxy,
        })
    return pd.DataFrame(rows)


def simulate_race_strategy(
    model,
    df: pd.DataFrame,
    current_lap: int,
    current_tyre_age: int,
    current_compound: str,
    speed: float,
    track_temp: float,
    race_laps: int,
    pit_lap: Optional[int],
    new_compound: Optional[str] = None,
    driver: str | None = None,
    air_temp: float = 25.0,
    humidity: float = 50.0,
    rainfall: float = 0.0,
    wind_speed: float = 0.0,
    pit_loss: float = DEFAULT_PIT_LOSS,
    strategy: str = "BALANCED",
) -> pd.DataFrame:
    """Return projected lap-by-lap race time for a one-stop candidate."""
    if current_lap >= race_laps:
        return pd.DataFrame()

    current_compound = current_compound.upper()
    new_compound = (new_compound or current_compound).upper()
    pace_reference = estimate_baseline_pace(df, "MEDIUM", driver)

    if pit_lap is None or pit_lap > race_laps:
        first_end = race_laps
    else:
        first_end = max(current_lap, pit_lap)

    first = simulate_stint(
        model, current_lap, current_tyre_age, current_compound, speed, track_temp,
        first_end, race_laps, air_temp, humidity, rainfall, wind_speed,
        pace_reference, "MEDIUM", strategy,
    )

    parts = [first]
    if pit_lap is not None and pit_lap < race_laps:
        sc, vsc = _track_status_for_lap(df, pit_lap)
        effective_loss = DEFAULT_SC_PIT_LOSS if sc else DEFAULT_VSC_PIT_LOSS if vsc else pit_loss
        second = simulate_stint(
            model, pit_lap, 1, new_compound, speed, track_temp,
            race_laps, race_laps, air_temp, humidity, rainfall, wind_speed,
            pace_reference, "MEDIUM", strategy,
        )
        second["pit_loss"] = 0.0
        if not second.empty:
            second.iloc[0, second.columns.get_loc("pit_loss")] = effective_loss
        parts.append(second)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not result.empty:
        result["pit_lap"] = pit_lap
        result["new_compound"] = new_compound if pit_lap is not None and pit_lap < race_laps else None
        result["cumulative_time"] = result["projected_lap_time"].cumsum() + result.get("pit_loss", 0.0).fillna(0.0).cumsum()
    return result


def compare_pit_windows(
    model,
    df: pd.DataFrame,
    current_lap: int,
    current_tyre_age: int,
    current_compound: str,
    speed: float,
    track_temp: float,
    race_laps: int,
    pit_laps: Iterable[int],
    new_compound: str = "HARD",
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for lap in pit_laps:
        if lap <= current_lap or lap >= race_laps:
            continue
        sim = simulate_race_strategy(
            model, df, current_lap, current_tyre_age, current_compound,
            speed, track_temp, race_laps, lap, new_compound, **kwargs
        )
        if sim.empty:
            continue
        rows.append({
            "pit_lap": lap,
            "new_compound": new_compound.upper(),
            "total_projected_time": float(sim["cumulative_time"].iloc[-1]),
            "pit_loss": float(sim.get("pit_loss", pd.Series([0.0])).sum()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        best = out["total_projected_time"].min()
        out["delta_to_best"] = out["total_projected_time"] - best
        out = out.sort_values("total_projected_time").reset_index(drop=True)
    return out


def detect_undercut_overcut(window_df: pd.DataFrame, current_lap: int, tolerance: float = 0.75) -> Dict[str, object]:
    if window_df.empty or len(window_df) < 2:
        return {"signal": "NONE", "strength": 0.0, "message": "Not enough pit-window scenarios."}
    ordered = window_df.sort_values("pit_lap")
    best_row = ordered.iloc[0]
    late = ordered.iloc[-1]
    early_candidates = ordered.iloc[: max(1, len(ordered) // 2)]
    late_candidates = ordered.iloc[max(0, len(ordered) // 2):]
    early_best = early_candidates.iloc[0]
    late_best = late_candidates.iloc[0]

    if late_best["total_projected_time"] + tolerance < early_best["total_projected_time"]:
        return {
            "signal": "OVERCUT",
            "strength": round(float(early_best["total_projected_time"] - late_best["total_projected_time"]), 2),
            "message": f"Later stop near Lap {int(late_best['pit_lap'])} projects faster by preserving the current tyre longer.",
        }
    if early_best["total_projected_time"] + tolerance < late_best["total_projected_time"]:
        return {
            "signal": "UNDERCUT",
            "strength": round(float(late_best["total_projected_time"] - early_best["total_projected_time"]), 2),
            "message": f"Earlier stop near Lap {int(early_best['pit_lap'])} projects faster on the fresh tyre.",
        }
    return {
        "signal": "NEUTRAL",
        "strength": round(float(abs(early_best["total_projected_time"] - late_best["total_projected_time"])), 2),
        "message": "Early and late pit options are within the strategy tolerance.",
    }


def monte_carlo_strategy(
    window_df: pd.DataFrame,
    simulations: int = 500,
    uncertainty_seconds: float = 0.35,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate probability each pit lap is optimal under pace uncertainty."""
    if window_df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    wins = {int(lap): 0 for lap in window_df["pit_lap"]}
    totals = {int(lap): [] for lap in window_df["pit_lap"]}
    for _ in range(simulations):
        sampled = {}
        for _, row in window_df.iterrows():
            lap = int(row["pit_lap"])
            sampled[lap] = float(row["total_projected_time"]) + rng.normal(0, uncertainty_seconds)
            totals[lap].append(sampled[lap])
        winner = min(sampled, key=sampled.get)
        wins[winner] += 1
    rows = []
    for lap in wins:
        rows.append({
            "pit_lap": lap,
            "win_probability": wins[lap] / simulations,
            "mean_projected_time": float(np.mean(totals[lap])),
        })
    return pd.DataFrame(rows).sort_values("win_probability", ascending=False).reset_index(drop=True)


def strategy_recommendation(window_df: pd.DataFrame, mc_df: pd.DataFrame | None = None) -> Dict[str, object]:
    if window_df.empty:
        return {"pit_lap": None, "confidence": 0.0, "message": "No valid strategy scenarios."}
    best = window_df.iloc[0]
    confidence = 0.55
    if mc_df is not None and not mc_df.empty:
        top = mc_df.iloc[0]
        confidence = min(0.95, max(0.50, float(top["win_probability"])))
        pit_lap = int(top["pit_lap"])
    else:
        pit_lap = int(best["pit_lap"])
    delta = float(best["delta_to_best"])
    if delta <= 0.5:
        message = f"Pit around Lap {pit_lap}; the window is tight and timing flexibility is high."
    elif delta <= 1.5:
        message = f"Pit around Lap {pit_lap}; nearby laps remain viable with a modest time penalty."
    else:
        message = f"Lap {pit_lap} is the clearest strategic optimum in the simulated window."
    return {"pit_lap": pit_lap, "confidence": confidence, "message": message}
