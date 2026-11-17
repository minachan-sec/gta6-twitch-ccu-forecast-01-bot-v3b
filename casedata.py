"""Case parsing for the GTA VI Twitch forecast entry.

Reads a makefun/twitch-ccu-case/v1 case object and turns it into the plain
records the forecaster needs: one target (hour-1 print, unlock class, diurnal
amplitude) and one record per usable comparable launch (hour-1 print, log
ratios for hours 2..24, unlock class, tier, diurnal amplitude).

Everything here uses only data present in the case file itself: the target's
history up to the end of hour 1, and the comparables' own hourly series.
"""

import math
import re
import statistics
from datetime import datetime, timezone


def parse_ts(text):
    """ISO 8601 timestamp (with a Z or an offset) -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_offset(ts, day_start):
    """Whole hours from day_start to ts (negative before the release day)."""
    return int((ts - day_start).total_seconds() // 3600)


def classify_unlock(text):
    """Rolling midnight, simultaneous global unlock, or anything else."""
    u = (text or "").lower()
    if "roll" in u or "midnight local" in u:
        return "rolling"
    if "simultaneous" in u or "global unlock" in u or "single global" in u:
        return "simultaneous"
    return "other"


def tier_of(text):
    """The comparable's tier tag, e.g. 'comparable', from its unlock text."""
    m = re.search(r"Tier:\s*([a-z-]+)", text or "")
    return m.group(1) if m else "none"


def diurnal_profile(hourly, day_start):
    """Mean log viewers by UTC hour (0..23) from pre-release hourly points.

    Only points before the release day count (hour offset < 0). Returns a
    24-vector on the log scale, circularly smoothed with a 3-point mean, or
    None when fewer than 12 distinct UTC hours are covered.
    """
    ds = parse_ts(day_start)
    by_hod = {}
    for p in hourly or []:
        v = p.get("viewers")
        if v is None or v < 1:
            continue
        ts = parse_ts(p["from"])
        if hour_offset(ts, ds) >= 0:
            continue
        by_hod.setdefault(ts.hour, []).append(math.log(v))
    if len(by_hod) < 12:
        return None
    means = {h: statistics.mean(vs) for h, vs in by_hod.items()}
    overall = statistics.mean(means.values())
    covered = sorted(means)
    prof = []
    for h in range(24):
        if h in means:
            prof.append(means[h])
            continue
        # Linear interpolation between the nearest covered neighbours
        # (circular), falling back to the overall mean when nothing is near.
        prev_h = max([c for c in covered if c < h], default=None)
        next_h = min([c for c in covered if c > h], default=None)
        if prev_h is not None and next_h is not None:
            frac = (h - prev_h) / (next_h - prev_h)
            prof.append(means[prev_h] * (1 - frac) + means[next_h] * frac)
        elif prev_h is not None or next_h is not None:
            # Wrap around the day boundary for the missing side.
            if prev_h is None:
                prev_h, next_h = next_h - 24, next_h
            else:
                prev_h, next_h = prev_h, prev_h + 24
            frac = (h - prev_h) / (next_h - prev_h) if next_h != prev_h else 0.0
            a = means[prev_h % 24]
            b = means[next_h % 24]
            prof.append(a * (1 - frac) + b * frac)
        else:
            prof.append(overall)
    return [(prof[(h - 1) % 24] + prof[h] + prof[(h + 1) % 24]) / 3.0 for h in range(24)]


def amplitude(profile):
    """Diurnal amplitude: the log-range of a diurnal profile, or None."""
    if profile is None:
        return None
    return max(profile) - min(profile)


def comparable_record(comp):
    """One comparable launch -> shape + features, or None if unusable.

    A launch is usable when its release day's hour 1 exists (hour offset 0,
    at least 1 viewer) and at least 18 of the 23 later hours have a print.
    The GTA VI category's own history series has no release day and drops
    out here, as do launches with a mostly missing release day.
    """
    ds = parse_ts(comp["day_start"])
    by_hour = {}
    for p in comp.get("hourly") or []:
        v = p.get("viewers")
        if v is None or not isinstance(v, (int, float)):
            continue
        try:
            h = hour_offset(parse_ts(p["from"]), ds)
        except (ValueError, TypeError, KeyError):
            continue
        if -400 <= h < 96:
            by_hour[h] = v
    if 0 not in by_hour or by_hour[0] < 1:
        return None
    v1 = by_hour[0]
    ratios = []
    for h in range(1, 24):
        if h in by_hour and by_hour[h] >= 1:
            ratios.append(math.log(by_hour[h] / v1))
        else:
            ratios.append(None)
    if sum(1 for r in ratios if r is not None) < 18:
        return None
    unlock = comp.get("unlock") or ""
    return {
        "name": comp["name"],
        "v1": v1,
        "ratios": ratios,
        "tier": tier_of(unlock),
        "uclass": classify_unlock(unlock),
        "amp": amplitude(diurnal_profile(comp.get("hourly") or [], comp["day_start"])),
    }


def target_info(case):
    """The target launch: hour-1 print, unlock class, diurnal amplitude."""
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1, first["viewers"])
    prof = diurnal_profile(case.get("history_hourly") or [], case["day_start"])
    return {
        "p1": p1,
        "uclass": classify_unlock(case.get("notes") or ""),
        "amp": amplitude(prof),
    }


def load_pool(case):
    """All usable comparables from a case, in file order."""
    pool = []
    for comp in case.get("comparables") or []:
        rec = comparable_record(comp)
        if rec is not None:
            pool.append(rec)
    return pool
