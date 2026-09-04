"""Competitor-aware race intelligence for APEXIA Phase 3.

The module deliberately labels gaps and position projections as estimates when
only lap timing data is available. It does not invent live telemetry.
"""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

from .model import COMPOUND_ENCODING, forecast_degradation, predict_degradation


def _clean_driver(value: object) -> str:
    return str(value).strip().upper()


def _latest_lap_rows(df: pd.DataFrame, current_lap: int) -> pd.DataFrame:
    work = df[df["lap"] <= current_lap].copy()
    if work.empty:
        return pd.DataFrame()
    work = work.sort_values(["driver", "lap"])
    return work.groupby("driver", as_index=False).tail(1).copy()


def _cumulative_times(df: pd.DataFrame, current_lap: int) -> pd.Series:
    work = df[df["lap"] <= current_lap].copy()
    work = work.dropna(subset=["driver", "lap_time"])
    if work.empty:
        return pd.Series(dtype=float)
    # Exclude obvious pit/flag outliers using the same broad race-time filter
    # already applied by the data loader. This remains an estimate of race gap.
    return work.groupby("driver")["lap_time"].sum()


def build_competitor_snapshot(
    df: pd.DataFrame,
    current_lap: int,
    target_driver: Optional[str] = None,
) -> pd.DataFrame:
    """Create the current race-state snapshot for every driver.

    Gap is estimated from cumulative recorded lap time. When Position is
    available, it is retained as a reference, but no claim is made that this
    replaces official timing gaps.
    """
    latest = _latest_lap_rows(df, current_lap)
    if latest.empty:
        return pd.DataFrame()

    latest["driver"] = latest["driver"].map(_clean_driver)
    target = _clean_driver(target_driver) if target_driver else latest.iloc[0]["driver"]
    cumulative = _cumulative_times(df, current_lap)
    latest["cumulative_time"] = latest["driver"].map(cumulative)
    target_time = float(cumulative.get(target, np.nan))
    latest["estimated_gap"] = latest["cumulative_time"] - target_time
    latest.loc[latest["driver"] == target, "estimated_gap"] = 0.0

    # Lower cumulative time is estimated as being ahead.
    latest["estimated_ahead"] = latest["estimated_gap"] < 0
    latest["is_target"] = latest["driver"] == target
    latest["tyre_age"] = pd.to_numeric(latest["tyre_age"], errors="coerce").fillna(0).astype(int)
    latest["compound"] = latest["compound"].astype(str).str.upper()
    latest["speed"] = pd.to_numeric(latest.get("speed", 300), errors="coerce").fillna(300)
    latest["track_temp"] = pd.to_numeric(latest.get("track_temp", 30), errors="coerce").fillna(30)
    latest["air_temp"] = pd.to_numeric(latest.get("air_temp", 25), errors="coerce").fillna(25)
    latest["humidity"] = pd.to_numeric(latest.get("humidity", 50), errors="coerce").fillna(50)
    latest["rainfall"] = pd.to_numeric(latest.get("rainfall", 0), errors="coerce").fillna(0)
    latest["wind_speed"] = pd.to_numeric(latest.get("wind_speed", 0), errors="coerce").fillna(0)

    # Official Position is useful when available in FastF1 timing data.
    if "position" in latest.columns:
        latest["position"] = pd.to_numeric(latest["position"], errors="coerce")
    else:
        latest["position"] = np.nan

    latest["relative_position_est"] = latest["cumulative_time"].rank(method="min").astype(int)
    latest["target_driver"] = target
    return latest.sort_values("relative_position_est").reset_index(drop=True)


def _driver_pace(df: pd.DataFrame, driver: str, current_lap: int) -> float:
    work = df[(df["driver"].map(_clean_driver) == _clean_driver(driver)) & (df["lap"] <= current_lap)].copy()
    if work.empty:
        return float(pd.to_numeric(df["lap_time"], errors="coerce").median())
    # Recent laps are more representative of current race pace.
    recent = work.sort_values("lap").tail(5)
    return float(pd.to_numeric(recent["lap_time"], errors="coerce").median())


def _project_driver_laps(
    model,
    row: pd.Series,
    start_lap: int,
    end_lap: int,
    race_laps: int,
    strategy: str = "BALANCED",
    pit_lap: Optional[int] = None,
    new_compound: Optional[str] = None,
    pit_loss: float = 21.5,
) -> float:
    if end_lap < start_lap:
        return 0.0
    driver = _clean_driver(row["driver"])
    compound = str(row["compound"]).upper()
    tyre_age = int(row["tyre_age"])
    speed = float(row.get("speed", 300))
    track_temp = float(row.get("track_temp", 30))
    air_temp = float(row.get("air_temp", 25))
    humidity = float(row.get("humidity", 50))
    rainfall = float(row.get("rainfall", 0))
    wind_speed = float(row.get("wind_speed", 0))
    pace = _driver_pace(row.get("_df", pd.DataFrame()), driver, start_lap) if isinstance(row.get("_df"), pd.DataFrame) else np.nan
    if not np.isfinite(pace):
        pace = float(row.get("pace_reference", 93.0))

    total = 0.0
    for lap in range(start_lap, end_lap + 1):
        age = tyre_age + (lap - start_lap + 1)
        used_compound = compound
        if pit_lap is not None and lap > pit_lap:
            age = lap - pit_lap
            used_compound = str(new_compound or compound).upper()
        fuel_proxy = max(0.0, 1.0 - lap / max(float(race_laps), 1.0))
        deg = predict_degradation(
            model, age, used_compound, speed, track_temp,
            air_temp, humidity, rainfall, wind_speed, fuel_proxy
        )
        total += pace + deg
        if pit_lap is not None and lap == pit_lap:
            total += pit_loss
    return float(total)


def _row_for(snapshot: pd.DataFrame, driver: str) -> Optional[pd.Series]:
    hit = snapshot[snapshot["driver"] == _clean_driver(driver)]
    return None if hit.empty else hit.iloc[0]


def find_rivals(snapshot: pd.DataFrame, target_driver: str, max_gap: float = 8.0) -> Dict[str, Optional[dict]]:
    """Return nearest estimated car ahead and behind within the supplied gap."""
    if snapshot.empty:
        return {"ahead": None, "behind": None}
    target = _row_for(snapshot, target_driver)
    if target is None:
        return {"ahead": None, "behind": None}

    others = snapshot[snapshot["driver"] != target["driver"]].copy()
    # Negative gap means competitor's cumulative time is lower -> ahead.
    ahead = others[(others["estimated_gap"] < 0) & (others["estimated_gap"] >= -max_gap)].sort_values("estimated_gap", ascending=False)
    behind = others[(others["estimated_gap"] > 0) & (others["estimated_gap"] <= max_gap)].sort_values("estimated_gap")

    def pack(row):
        if row is None or len(row) == 0:
            return None
        r = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        return {
            "driver": r["driver"],
            "gap": abs(float(r["estimated_gap"])),
            "compound": r["compound"],
            "tyre_age": int(r["tyre_age"]),
            "position": None if pd.isna(r.get("position")) else int(r["position"]),
        }

    return {"ahead": pack(ahead), "behind": pack(behind)}


def competitor_strategy_analysis(
    model,
    df: pd.DataFrame,
    current_lap: int,
    race_laps: int,
    target_driver: str,
    target_new_compound: str = "HARD",
    pit_loss: float = 21.5,
    horizon: int = 6,
) -> pd.DataFrame:
    """Evaluate the target's next few laps against the nearest rivals.

    This is a tactical, short-horizon model. It estimates whether pitting in the
    next few laps improves the target's cumulative pace relative to each rival.
    """
    snapshot = build_competitor_snapshot(df, current_lap, target_driver)
    if snapshot.empty:
        return pd.DataFrame()
    target_row = _row_for(snapshot, target_driver)
    if target_row is None:
        return pd.DataFrame()

    target_row = target_row.copy()
    target_row["_df"] = df
    rivals = find_rivals(snapshot, target_driver, max_gap=10.0)
    rows = []
    for relation, rival_info in rivals.items():
        if not rival_info:
            continue
        rival_row = _row_for(snapshot, rival_info["driver"])
        if rival_row is None:
            continue
        rival_row = rival_row.copy()
        rival_row["_df"] = df
        rival_pace = _driver_pace(df, rival_row["driver"], current_lap)

        baseline_total = 0.0
        best_pit = None
        best_delta = None
        candidate_end = min(race_laps - 1, current_lap + horizon)
        for pit in range(current_lap + 1, candidate_end + 1):
            target_with_pit = _project_driver_laps(
                model, target_row, current_lap + 1, candidate_end, race_laps,
                pit_lap=pit, new_compound=target_new_compound, pit_loss=pit_loss,
            )
            rival_no_pit = _project_driver_laps(
                model, rival_row, current_lap + 1, candidate_end, race_laps,
            )
            # Current gap is paid first; lower relative time is better.
            relative = float(rival_info["gap"] + target_with_pit - rival_no_pit)
            if best_delta is None or relative < best_delta:
                best_delta, best_pit = relative, pit

        current_gap = float(rival_info["gap"])
        rival_age = int(rival_info["tyre_age"])
        target_age = int(target_row["tyre_age"])
        age_advantage = target_age - rival_age
        if relation == "ahead":
            # Positive means target could close the gap under the pit scenario.
            effect = current_gap - best_delta if best_delta is not None else 0.0
            if effect > 0.5:
                signal = "UNDERCUT OPPORTUNITY"
            elif age_advantage > 5:
                signal = "OVERCUT OPPORTUNITY"
            else:
                signal = "HOLD"
        else:
            # A nearby faster-tyre car behind increases the defensive pressure.
            effect = current_gap - best_delta if best_delta is not None else 0.0
            if current_gap < 2.0 and rival_age < target_age:
                signal = "DEFEND / CONSIDER PIT"
            elif effect > 0.5:
                signal = "PIT TO PROTECT"
            else:
                signal = "HOLD"

        rows.append({
            "relation": relation.upper(),
            "rival": rival_row["driver"],
            "gap_s": round(current_gap, 2),
            "target_tyre": target_row["compound"],
            "target_age": target_age,
            "rival_tyre": rival_row["compound"],
            "rival_age": rival_age,
            "best_response_lap": best_pit,
            "relative_projection_s": round(float(best_delta or 0.0), 2),
            "signal": signal,
        })
    return pd.DataFrame(rows)


def build_position_projection(
    model,
    df: pd.DataFrame,
    current_lap: int,
    race_laps: int,
    target_driver: str,
    pit_lap: int,
    new_compound: str,
    pit_loss: float = 21.5,
    horizon: int = 5,
) -> pd.DataFrame:
    """Project relative order for a short tactical horizon.

    The projection is based on lap-time sums and therefore is explicitly an
    estimate, not official classification timing.
    """
    snapshot = build_competitor_snapshot(df, current_lap, target_driver)
    if snapshot.empty:
        return pd.DataFrame()
    end_lap = min(race_laps, current_lap + horizon)
    rows = []
    for _, row in snapshot.iterrows():
        r = row.copy()
        r["_df"] = df
        pit = pit_lap if _clean_driver(r["driver"]) == _clean_driver(target_driver) else None
        total = _project_driver_laps(
            model, r, current_lap + 1, end_lap, race_laps,
            pit_lap=pit, new_compound=new_compound, pit_loss=pit_loss,
        )
        rows.append({"driver": r["driver"], "projected_time": total, "is_target": bool(r["is_target"])})
    result = pd.DataFrame(rows).sort_values("projected_time").reset_index(drop=True)
    result["projected_position"] = np.arange(1, len(result) + 1)
    return result


def race_engineer_call(
    snapshot: pd.DataFrame,
    tactical: pd.DataFrame,
    projection: pd.DataFrame,
    target_driver: str,
) -> Dict[str, object]:
    """Turn the tactical outputs into a concise race-engineer message."""
    if snapshot.empty:
        return {"signal": "NO DATA", "severity": "INFO", "message": "No competitor timing data is available."}

    target = _row_for(snapshot, target_driver)
    target_pos = int(target["relative_position_est"]) if target is not None else None
    target_tactical = tactical[tactical["signal"] != "HOLD"] if not tactical.empty else tactical

    if not target_tactical.empty:
        priority = ["UNDERCUT OPPORTUNITY", "DEFEND / CONSIDER PIT", "PIT TO PROTECT", "OVERCUT OPPORTUNITY"]
        chosen = None
        for p in priority:
            hit = target_tactical[target_tactical["signal"] == p]
            if not hit.empty:
                chosen = hit.iloc[0]
                break
        if chosen is not None:
            return {
                "signal": str(chosen["signal"]),
                "severity": "HIGH" if "UNDERCUT" in chosen["signal"] or "DEFEND" in chosen["signal"] else "MEDIUM",
                "message": f"{chosen['rival']} is {float(chosen['gap_s']):.1f}s {str(chosen['relation']).lower()}. Best tactical response is around Lap {int(chosen['best_response_lap'])}.",
            }

    if projection is not None and not projection.empty:
        hit = projection[projection["is_target"]]
        if not hit.empty:
            projected_pos = int(hit.iloc[0]["projected_position"])
            if target_pos is not None and projected_pos < target_pos:
                return {"signal": "POSITION GAIN", "severity": "MEDIUM", "message": f"The selected pit scenario projects a gain from P{target_pos} to P{projected_pos}."}
            if target_pos is not None and projected_pos > target_pos:
                return {"signal": "POSITION RISK", "severity": "MEDIUM", "message": f"The selected pit scenario projects a drop from P{target_pos} to P{projected_pos}."}

    return {"signal": "HOLD", "severity": "INFO", "message": "No strong competitor-triggered pit signal detected. Maintain the current plan and monitor the nearest cars."}
