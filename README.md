# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/junobebe2-ux/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/junobebe2-ux/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Trophy rarity forecaster for GTA VI — a program that forecasts every GTA VI PS5 trophy's Exophase earn rate a week after launch, for make.fun gta6-trophy-rarity-forecast-01.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## The model

A trophy's earn rate is `earners / players_tracked`, so the program models the two separately and divides: a log-time power-law fit for players, a per-trophy log-time fit for earners (damped by the headroom under each trophy's first-week ceiling), a refresh prior when the history is flat, and a spread calibrated from the observed refresh jumps. Story-chain monotonicity is enforced (Chapter 1 ≥ Chapter 2 ≥ … ≥ final mission; platinum ≤ all others).

---

<sub>Entered on [make.fun](https://make.fun/?ref=52m6pxck&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
