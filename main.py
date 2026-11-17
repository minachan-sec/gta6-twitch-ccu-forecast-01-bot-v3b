"""Entry point for the GTA VI Twitch forecast (v3b).

Reads every <id>.in case in the input directory and writes one <id>.out
forecast per case to the output directory:
  {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ..., {"hour": 24, ...}]}
23 rows, hours 2..24, five non-decreasing quantiles each, all finite,
above 0 and at most 1e9 viewers.

Directories come from MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR when set,
else /input / /output (the grader's container leaves both unset).

Run locally with:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader runs `python3 -P -s /bot/main.py`, which leaves /bot off
# sys.path: put this file's directory back first so `import model` and
# `import casedata` find the modules shipped beside it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import casedata
import model

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
MAX_VIEWERS = 1_000_000_000


def round2(x):
    """x rounded to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def clean_row(quantiles):
    """Clamp one row of five quantiles to the contest's output contract:
    finite, above 0, at most 1e9, non-decreasing, rounded to 2 decimals."""
    out = []
    for q in quantiles:
        if not math.isfinite(q) or q <= 0:
            q = 1.0
        q = min(q, MAX_VIEWERS)
        out.append(round2(q))
    for j in range(1, 5):
        out[j] = max(out[j], out[j - 1])
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    target = casedata.target_info(case)
    pool = casedata.load_pool(case)
    rows = model.forecast(target["p1"], target["uclass"], target["amp"], pool)
    # Map each forecast row to its model row by hour number, not by position,
    # so the output stays correct even if the case lists hours out of order.
    by_hour = {h: rows[h - 2] for h in range(2, 25)}
    hours = [{"hour": row["hour"], "q": clean_row(by_hour[row["hour"]])}
             for row in sorted(case["forecast"], key=lambda r: r["hour"])]
    return {"hours": hours}


def fallback_case(case):
    """Persistence forecast used when a case cannot be modelled at all:
    every hour stays at hour 1's print with a widening spread. Better than
    an invalid case (which scores as a degenerate all-equal forecast)."""
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    p1 = max(1, first["viewers"]) if first else 1
    hours = []
    for row in sorted(case["forecast"], key=lambda r: r["hour"]):
        sig = 0.15 * math.sqrt(row["hour"] - 1)
        hours.append({"hour": row["hour"],
                      "q": clean_row([p1 * math.exp(z * sig) for z in model.Z])})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # One bad case must not take the whole run down: fall back to
            # persistence for it and keep the other cases' forecasts.
            forecast = fallback_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
