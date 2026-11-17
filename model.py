"""The v3b forecaster: kernel-weighted comparable shapes + per-hour WLS,
with a conditional bias correction, a diurnal-amplitude dip adjustment and
an early-hour rolling-unlock adjustment.

Model, per forecast hour i (0..22, task hours 2..24):
  median = (1 - W_REG) * kernel_median + W_REG * wls_median + bias[i]
  spread = asymmetric parametric spread from the kernel-weighted residual
           scatter above/below the median, smoothed across hours, scaled
           by SPREAD_SCALE, floored at FLOOR_SIG * sqrt(i + 1).
  quantiles = median + z_tau * sigma (z from the standard normal).

The bias correction is estimated by leave-one-out over the comparable pool:
each comparable is forecast by the base model from the rest, the residuals
are kernel-weighted by similarity to the target, shrunk by their t-statistic
and capped. The dip adjustment deepens (or shallows) the midday hours
10..14 for targets whose pre-release diurnal amplitude is above (or below)
the pool's, shrunk for small launches where the amplitude is noise. The
early adjustment lowers hours 2..4 for rolling-midnight launches, whose
first hours only hold the earliest time zones.

All parameters were tuned on the four practice cases with leave-one-out
over the comparable pool as a guardrail (practice 0.139, LOO 0.402,
clean-tier BIG-regime LOO 0.152). Deterministic; no randomness anywhere.
"""

import math
import statistics

# --- tuned parameters ---
KERNEL_BW = 1.0        # Gaussian kernel bandwidth on log hour-1 viewers
UNLOCK_BOOST = 3.0     # weight multiplier for same unlock class
TIER_DOWN = 0.3        # weight multiplier for non-'comparable' tiers
W_REG = 0.9            # weight on the WLS median (rest on kernel median)
SPREAD_SCALE = 0.6     # scale on the empirical spread
FLOOR_SIG = 0.05       # spread floor, in log units per sqrt hour
SMOOTH_SIG = 0.5       # smoothing strength for the spread vectors
BIAS_CAP = 0.10        # max |bias correction| per hour, in log units
BIAS_BW = 1.0          # kernel bandwidth for the bias correction
DIP_GAIN = 0.5         # diurnal-amplitude dip adjustment gain
DIP_HOURS = (8, 9, 10, 11, 12, 13)   # indices of task hours 10..14
EARLY_GAIN = 0.1       # early-hour rolling adjustment, in log units
EARLY_HOURS = (0, 1, 2)              # indices of task hours 2..4
DIP_SIZE_LO = 5000     # dip reliability: ~0 gain below this hour-1 size
DIP_SIZE_HI = 100000   # dip reliability: full gain above this hour-1 size
WLS_CLAMP = 1.5        # WLS median may stray at most this far from kernel

Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
NHOURS = 23


def smooth(vec, strength=0.5):
    """One pass of 3-point smoothing (endpoints use their single neighbour)."""
    out = []
    for i, v in enumerate(vec):
        lo = vec[i - 1] if i > 0 else v
        hi = vec[i + 1] if i < len(vec) - 1 else v
        out.append((1 - strength) * v + strength * (lo + v + hi) / 3.0)
    return out


def kernel_weights(log_p1, uclass, pool):
    """Gaussian weights on log hour-1 size, boosted for same unlock class,
    downweighted for non-'comparable' tiers."""
    ws = []
    for o in pool:
        w = math.exp(-((log_p1 - math.log(o["v1"])) ** 2) / (2 * KERNEL_BW ** 2))
        if o["uclass"] == uclass:
            w *= UNLOCK_BOOST
        if o["tier"] != "comparable":
            w *= TIER_DOWN
        ws.append(w)
    return ws


def weighted_quantiles(pts, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Weighted quantiles of [(value, weight), ...]; first value whose
    cumulative weight reaches q * total weight."""
    pts = sorted(pts)
    total = sum(w for _, w in pts)
    out = []
    for q in qs:
        acc = 0.0
        val = pts[-1][0]
        for r, w in pts:
            acc += w
            if acc >= q * total:
                val = r
                break
        out.append(val)
    return out


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 3:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-9:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return (my - b * mx, b)


def core_components(log_p1, uclass, pool):
    """Per hour: kernel median, WLS median, weighted residual points."""
    ws = kernel_weights(log_p1, uclass, pool)
    comps = []
    for i in range(NHOURS):
        pts = [(o["ratios"][i], w) for o, w in zip(pool, ws)
               if o["ratios"][i] is not None]
        if not pts:
            comps.append(None)
            continue
        med_k = weighted_quantiles(pts)[2]
        xs = [math.log(o["v1"]) for o in pool if o["ratios"][i] is not None]
        ys = [o["ratios"][i] for o in pool if o["ratios"][i] is not None]
        w2 = [w for o, w in zip(pool, ws) if o["ratios"][i] is not None]
        fit = wls_line(xs, ys, w2)
        med_r = fit[0] + fit[1] * log_p1 if fit else med_k
        med_r = max(med_k - WLS_CLAMP, min(med_k + WLS_CLAMP, med_r))
        comps.append({"pts": pts, "med_k": med_k, "med_r": med_r})
    return comps


def build_rows(p1, comps, bias):
    """Assemble 23 rows of 5 quantiles (absolute viewers) from components
    plus a per-hour bias vector."""
    meds, sig_up, sig_dn = [], [], []
    for i, c in enumerate(comps):
        if c is None:
            meds.append(0.0)
            sig_up.append(FLOOR_SIG * math.sqrt(i + 1))
            sig_dn.append(FLOOR_SIG * math.sqrt(i + 1))
            continue
        med = (1 - W_REG) * c["med_k"] + W_REG * c["med_r"] + bias[i]
        meds.append(med)
        up = [(r - med, w) for r, w in c["pts"] if r >= med]
        dn = [(r - med, w) for r, w in c["pts"] if r < med]

        def wstd(seg):
            s = sum(w for _, w in seg)
            if s <= 0:
                return None
            return math.sqrt(sum(w * d * d for d, w in seg) / s)

        su, sd = wstd(up), wstd(dn)
        if su is None:
            su = sd if sd is not None else FLOOR_SIG * math.sqrt(i + 1)
        if sd is None:
            sd = su
        sig_up.append(su)
        sig_dn.append(sd)
    sig_up = smooth(sig_up, SMOOTH_SIG)
    sig_dn = smooth(sig_dn, SMOOTH_SIG)
    rows = []
    for i in range(NHOURS):
        fl = FLOOR_SIG * math.sqrt(i + 1)
        su = max(fl, sig_up[i] * SPREAD_SCALE)
        sd = max(fl, sig_dn[i] * SPREAD_SCALE)
        qs = [meds[i] - 1.645 * sd, meds[i] - 0.674 * sd, meds[i],
              meds[i] + 0.674 * su, meds[i] + 1.645 * su]
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append([p1 * math.exp(q) for q in qs])
    return rows


def base_forecast(p1, uclass, pool):
    """The base model (no bias correction) for one target."""
    comps = core_components(math.log(p1), uclass, pool)
    return build_rows(p1, comps, [0.0] * NHOURS)


def conditional_bias(p1, uclass, pool):
    """Per-hour bias from leave-one-out residuals of the pool, kernel
    weighted by similarity to the target, shrunk by t-statistic, capped
    and smoothed. Returns a 23-vector of log corrections."""
    log_p1 = math.log(p1)
    if len(pool) < 8:
        # Too few comparables for a stable leave-one-out estimate.
        return [0.0] * NHOURS
    resid = [[] for _ in range(NHOURS)]
    for t in pool:
        sub = [o for o in pool if o["name"] != t["name"]]
        if len(sub) < 3:
            continue
        rows = base_forecast(t["v1"], t["uclass"], sub)
        w = math.exp(-((log_p1 - math.log(t["v1"])) ** 2) / (2 * BIAS_BW ** 2))
        if t["uclass"] == uclass:
            w *= UNLOCK_BOOST
        for i in range(NHOURS):
            if t["ratios"][i] is None:
                continue
            resid[i].append((t["ratios"][i] - math.log(rows[i][2] / t["v1"]), w))
    bias = []
    for i in range(NHOURS):
        pts = resid[i]
        if len(pts) < 5:
            bias.append(0.0)
            continue
        sw = sum(w for _, w in pts)
        if sw <= 0:
            bias.append(0.0)
            continue
        mean = sum(r * w for r, w in pts) / sw
        var = sum(w * (r - mean) ** 2 for r, w in pts) / sw
        n_eff = sw * sw / sum(w * w for _, w in pts)
        se = math.sqrt(var / max(n_eff, 1.0))
        t = abs(mean) / se if se > 0 else 0.0
        shrink = t * t / (t * t + 1.0)
        bias.append(max(-BIAS_CAP, min(BIAS_CAP, mean * shrink)))
    return smooth(bias, 0.5)


def forecast(p1, uclass, amp, pool):
    """Full v3b forecast: 23 rows of 5 quantiles (absolute viewers)."""
    if not pool:
        # No usable comparables: persistence baseline with a widening spread.
        rows = []
        for i in range(NHOURS):
            sig = 0.15 * math.sqrt(i + 1)
            rows.append([p1 * math.exp(z * sig) for z in Z])
        return rows
    bias = conditional_bias(p1, uclass, pool)
    comps = core_components(math.log(p1), uclass, pool)
    rows = build_rows(p1, comps, bias)
    # Diurnal-amplitude dip adjustment (midday hours 10..14).
    amps = [o["amp"] for o in pool
            if o.get("amp") is not None and o["amp"] > 0]
    if amp is not None and amp > 0 and amps:
        mean_log_amp = statistics.mean(math.log(a) for a in amps)
        rel = (math.log(p1) - math.log(DIP_SIZE_LO)) / (
            math.log(DIP_SIZE_HI) - math.log(DIP_SIZE_LO))
        rel = max(0.0, min(1.0, rel))
        adj = DIP_GAIN * rel * (math.log(amp) - mean_log_amp)
        factor = math.exp(-adj)
        rows = [[q * factor if i in DIP_HOURS else q for q in r]
                for i, r in enumerate(rows)]
    # Early-hour rolling adjustment (hours 2..4, rolling launches only).
    if uclass == "rolling":
        factor = math.exp(-EARLY_GAIN)
        rows = [[q * factor if i in EARLY_HOURS else q for q in r]
                for i, r in enumerate(rows)]
    return rows
