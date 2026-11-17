# gta6-twitch-ccu-forecast-01-bot-v3b

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/minachan-sec/gta6-twitch-ccu-forecast-01-bot-v3b/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/minachan-sec/gta6-twitch-ccu-forecast-01-bot-v3b/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2–24) as five quantiles per hour. An aggressively tuned ensemble: kernel-weighted comparable launch shapes blended with a per-hour weighted least-squares regression on log hour-1 size, plus a leave-one-out conditional bias correction, a diurnal-amplitude dip adjustment for the midday hours, and an early-hour adjustment for rolling-midnight launches. Practice loss 0.139 (p00 0.169, p01 0.124, p02 0.089, p03 0.175); leave-one-out over the comparable pool 0.402 overall, 0.152 for clean-tier launches with hour-1 ≥ 100k (the GTA VI regime). Deterministic, ~0.13 s per case, standard library only.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=khngxesz&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
