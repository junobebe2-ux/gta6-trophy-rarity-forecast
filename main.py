"""gta6-trophy-rarity-forecast-01 entry: forecast every GTA VI PS5 trophy's
Exophase earn rate at the case's target, as five quantiles a trophy.

Model (stdlib only, deterministic, no network):

  Earners and players, separately.  A trophy's rate is earners / players.
  Both are counts, so both only grow; the rate falls when players join
  faster than earners do (Exophase refreshes in jumps: a refresh adds a
  batch of new players who hold few trophies, so rates drop) and rises
  when the players already tracked progress through the game (early
  story trophies are earned by most players within days).

  1. Players: least-squares line of ln(players) on ln(1 + hours since
     the release, 2026-11-19T00:00Z), over all the case's reads, read at
     the target.  Players only grow, so the slope is clamped to >= 0.
     (A recent-24h slope was tried for the players projection and
     rejected: on the synthetic final-like case it raised 34 of 55
     medians, because lowering only the players slope doubles the
     implied rate growth; the whole-history power law is kept.)
  2. Earners, per trophy: earners = rate * players / 100 in each read.
     Least-squares line of ln(earners + 0.5) on ln(1 + hours), read at
     the target.  Earners only grow, so the slope is clamped to >= 0
     and the projection to >= the last read's earners and <= the
     projected players.
  3. Median rate = 100 * earners(target) / players(target), capped at
     a ceiling for the trophy: the share of players who will ever
     hold it in the first week.  The ceiling is the lower of two
     bounds.  (a) An absolute cap from the trophy's place in the
     game, recalibrated down for realistic week-one playtime in a
     30+ hour story (intro 100%, Chapter 1 72%, Chapter 2 58%,
     Chapter 3 47%, Chapter 4 37%, final mission 14%, platinum 1%,
     100% Completion 2%, other gold 25%, silver 25%, bronze 30%).
     Most players will not finish a 30+ hour story in a week, and
     the platinum needs essentially everything, which takes months.
     (b) A data-driven chain bound: for each step of the story
     chain, the ceiling is the previous chapter's ceiling times the
     chapter-to-chapter completion ratio at the target, where the
     ratio is fitted from the history (its trend against log-time,
     read at the target, so a ratio still rising as players catch
     up is allowed to rise, and a flat one stays flat).  The
     platinum is also bounded by the 100% Completion ceiling, and
     100% Completion by the final mission's ceiling times their
     observed ratio.  The ceiling is never below the last rate
     times 1.1, so a trophy already past its cap is not forced
     down.  Without a ceiling the power-law earners fit
     extrapolates launch week's explosive growth straight to ~100%
     for every early trophy, which no ceiling allows.
  4. Flat history (no read differs from another, as on launch night):
     the fits are flat, so a refresh prior is applied instead.  A
     refresh is assumed in the horizon.  It adds a batch of new players
     (about 3,000 a refresh, the size of the first refresh the house
     saw: 638 -> 3,868 players) who have barely played, so they hold
     a trophy at only a small fraction k of the old players' rate --
     k grows with how early the trophy is (its current rate, the best
     proxy for earliness) and is ~0.07 on average, which is what the
     first refresh's totals imply (non-intro earners 305 -> 408 while
     players went 638 -> 3,868).  The median is then
     (old earners + new earners) / (old players + new players), so
     most rates fall to roughly a fifth of where they stood, while
     the 100% intro trophy stays 100%.
  5. Monotonicity the game forces: the platinum (earned by holding
     every other trophy) is <= every other trophy's median; the story
     chain (Gone Fishing, Chapter 1..4, final mission) is
     non-increasing in that order; 100% Completion is <= the final
     mission.  Enforced on the medians by taking running minima.
  6. Spread, in logit units: the random-walk baseline
     0.5 * sqrt(horizon in days), combined with a refresh-jump term
     and the earners fit's residual scatter.  The refresh term is
     calibrated from the history where it has real refreshes: the
     pooled RMS of the logit change at each read where the players
     count moved, clamped to [0.3, 0.8].  Where the history is flat
     (no refresh at all, as on launch night) it falls back to a 0.5
     floor.  The 90-minute practice horizon gets a spread several
     times the baseline's, because a refresh in it moves every rate
     by ~0.3 logit; the 4.5-day final horizon is dominated by the
     random-walk term.
  7. Quantiles: q_tau = inv_logit(logit(median) + z_tau * sigma),
     z = (-1.645, -0.674, 0, 0.674, 1.645), rounded to 4 decimals.

Rates are clamped to [0.01, 99.99] before any logit.  A trophy with
no rate in a read is skipped in that read's fit.  A case with one
read, one distinct players value, or all-null rates for a trophy
falls back to the last rate with the refresh prior; it never crashes.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release of the game; the log-time covariate is hours since this instant.
RELEASE = "2026-11-19T00:00:00Z"
# A refresh is assumed about this often (Exophase refreshes a game's
# counts hours apart, not continuously).
REFRESH_HOURS = 4.0
# Logit spread floor: the refresh jumps the history has not shown yet.
# Used only where the history has no real refresh to calibrate from.
SPREAD_FLOOR = 0.5
# Bounds on the calibrated refresh volatility (pooled RMS logit change
# at refreshes).  Below 0.3 the spread would claim more precision than
# any refresh ever showed; above 0.8 one extreme launch refresh would
# blow it out.
SPREAD_CAL_MIN = 0.3
SPREAD_CAL_MAX = 0.8
# New players a refresh adds, on the launch-week prior (the first
# refresh the house saw added 3,230).  A refresh is assumed about
# every REFRESH_HOURS.
NEW_PLAYERS_PER_REFRESH = 3000.0
# Random-walk spread per sqrt day, as the house baseline uses it.
SIGMA_RW = 0.5

# Absolute ceiling caps (fraction 0..1) on the share of players who
# will hold a trophy in the first week, from its place in the game.
# Recalibrated down for realistic week-one playtime in a 30+ hour
# story: most players will not finish it in a week, and the platinum
# needs essentially everything, which takes months, not a week.
ABS_CAP = {
    "intro": 1.0,
    "ch1": 0.72,
    "ch2": 0.58,
    "ch3": 0.47,
    "ch4": 0.37,
    "final": 0.14,
    "platinum": 0.01,
    "completion": 0.02,
    "gold": 0.25,
    "silver": 0.25,
    "bronze": 0.30,
}


def parse_time(s):
    """ISO 8601 UTC timestamp -> epoch seconds (no datetime import cost
    issues; fromisoformat handles the Z suffix in 3.11+)."""
    return _dt_fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _dt_fromisoformat(s):
    import datetime
    return datetime.datetime.fromisoformat(s)


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as the starter does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def linfit(xs, ys):
    """Least-squares line y = a + b*x.  Returns (a, b, resid_var).
    With < 2 points, or no x variance, returns (mean y, 0, 0)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    my = sum(ys) / n
    if n < 2:
        return my, 0.0, 0.0
    mx = sum(xs) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n
    return a, b, resid


def classify_trophy(trophy):
    """Which ceiling class a trophy belongs to: intro, ch1..ch4,
    final, platinum, completion, or its tier (gold/silver/bronze)."""
    d = (trophy.get("description") or "").lower()
    tier = trophy.get("tier") or ""
    if "gone fishing" in d:
        return "intro"
    if "chapter 1" in d:
        return "ch1"
    if "chapter 2" in d:
        return "ch2"
    if "chapter 3" in d:
        return "ch3"
    if "chapter 4" in d:
        return "ch4"
    if "final mission" in d and tier == "gold":
        return "final"
    if tier == "platinum":
        return "platinum"
    if "100% completion" in d:
        return "completion"
    if tier == "gold":
        return "gold"
    if tier == "silver":
        return "silver"
    return "bronze"


def ratio_target(reads, idx_next, idx_prev, u_target):
    """Chapter-to-chapter completion ratio at the target, from the
    history: rate(next) / rate(prev), fitted against ln(1 + hours)
    and read at the target.  Only reads after the first players
    change are used -- the flat launch-night reads have all rates
    equal, so their ratio is meaningless.  The fitted ratio is
    clamped to [last ratio, 0.95]: a ratio only rises as players
    progress through the chain, and it never reaches 1 (some
    players stop after every chapter).  Returns None when the
    history carries no usable pair."""
    first_p = None
    for t, h in reads:
        first_p = h.get("players_tracked")
        break
    xs, ys = [], []
    for t, h in reads:
        if h.get("players_tracked") == first_p:
            continue
        rates = h.get("rates") or []
        a = rates[idx_prev] if idx_prev < len(rates) else None
        b = rates[idx_next] if idx_next < len(rates) else None
        if a is not None and b is not None and a > 0:
            xs.append(math.log1p(t))
            ys.append(b / a)
    if len(ys) < 2:
        return ys[-1] if ys else None
    a_fit, b_fit, _ = linfit(xs, ys)
    val = a_fit + b_fit * u_target
    return min(0.95, max(ys[-1], val))


def refresh_volatility(reads):
    """Pooled RMS of the logit change at each read where the players
    count moved, across every trophy.  That is how far a refresh
    actually moves the rates, which is what the spread's refresh
    term should reflect.  Returns None when the history has no
    such read (a flat history)."""
    total = 0.0
    count = 0
    prev = None
    for t, h in reads:
        if prev is not None and h.get("players_tracked") != prev.get("players_tracked"):
            rates = h.get("rates") or []
            prev_rates = prev.get("rates") or []
            for i in range(min(len(rates), len(prev_rates))):
                a = prev_rates[i]
                b = rates[i]
                if a is None or b is None:
                    continue
                d = logit(b) - logit(a)
                total += d * d
                count += 1
        prev = h
    if count == 0:
        return None
    return math.sqrt(total / count)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0

    t0 = parse_time(RELEASE)
    # Times in hours since release; u = ln(1 + t) is the covariate.
    reads = []
    for h in history:
        t = (parse_time(h["at"]) - t0) / 3600.0
        if t < 0:
            t = 0.0
        reads.append((t, h))
    t_target = (parse_time(case["target"]) - t0) / 3600.0
    if t_target < 0:
        t_target = 0.0
    u_target = math.log1p(t_target)

    last = reads[-1][1]
    p_last = last["players_tracked"] or 0
    if p_last <= 0:
        # No usable players count: fall back to the last rates.
        p_last = 1

    # ---- players ----
    xs_p, ys_p = [], []
    for t, h in reads:
        p = h.get("players_tracked")
        if p and p > 0:
            xs_p.append(math.log1p(t))
            ys_p.append(math.log(p))
    a_p, b_p, _ = linfit(xs_p, ys_p)
    flat_players = len(set(ys_p)) <= 1
    b_p = min(max(b_p, 0.0), 3.0)
    # Project players from the last read, not the fit's intercept.
    u_last_p = math.log1p(reads[-1][0])
    p_hat = float(p_last) * math.exp(b_p * (u_target - u_last_p))
    p_hat = min(max(p_hat, float(p_last)), float(p_last) * 50.0)

    # ---- last rates and earners ----
    # The last rate is the most recent non-null read for the trophy,
    # not just the last read's value (a read can lack a trophy).
    # We keep the players count from that same read, so the earners
    # figure (rate * players) is from one moment, not two.
    last_rates = []
    last_players = []
    for i in range(n):
        r = None
        pp = p_last
        for t, h in reversed(reads):
            rates = h.get("rates") or []
            v = rates[i] if i < len(rates) else None
            if v is not None:
                r = v
                pp = h.get("players_tracked") or p_last
                break
        last_rates.append(r if r is not None else 0.0)
        last_players.append(pp if pp and pp > 0 else p_last)

    # ---- ceilings ----
    # Two bounds, the lower wins.  (a) An absolute cap from the
    # trophy's place in the game (ABS_CAP), recalibrated down for
    # realistic week-one playtime.  (b) A data-driven chain bound:
    # for each step of the story chain, the previous chapter's
    # ceiling times the chapter-to-chapter completion ratio at the
    # target, fitted from the history.  The platinum is bounded by
    # the 100% Completion ceiling, and 100% Completion by the final
    # mission's ceiling times their observed ratio.  The ceiling is
    # never below the last rate times 1.1.
    classes = [classify_trophy(t) for t in trophies]
    by_id = {t["id"]: idx for idx, t in enumerate(trophies)}
    ceilings = []
    for t, r in zip(trophies, last_rates):
        ceilings.append(max(ABS_CAP[classify_trophy(t)], (r / 100.0) * 1.1))
    chain = story_chain(trophies)
    if len(chain) >= 2:
        # Walk the chain in order; each step's ceiling is the
        # previous chapter's ceiling times the ratio at the target.
        prev_idx = chain[0]
        ceilings[prev_idx] = min(ceilings[prev_idx], ABS_CAP[classes[prev_idx]])
        for idx in chain[1:]:
            ratio = ratio_target(reads, idx, prev_idx, u_target)
            if ratio is not None:
                ceilings[idx] = min(ceilings[idx], ceilings[prev_idx] * ratio)
            ceilings[idx] = max(ceilings[idx], (last_rates[idx] / 100.0) * 1.1)
            prev_idx = idx
    # 100% Completion is bounded by the final mission's ceiling
    # times their observed ratio; the platinum by the 100%
    # Completion ceiling (it needs every trophy, 100% included).
    if 14 in by_id and 7 in by_id:
        ratio = ratio_target(reads, by_id[14], by_id[7], u_target)
        if ratio is not None:
            ceilings[by_id[14]] = min(
                ceilings[by_id[14]], ceilings[by_id[7]] * ratio)
        ceilings[by_id[14]] = max(
            ceilings[by_id[14]], (last_rates[by_id[14]] / 100.0) * 1.1)
    if 1 in by_id and 14 in by_id:
        ceilings[by_id[1]] = min(ceilings[by_id[1]], ceilings[by_id[14]])
        ceilings[by_id[1]] = max(
            ceilings[by_id[1]], (last_rates[by_id[1]] / 100.0) * 1.1)
    # Clamp every ceiling to [0, 1] (the 1.1 floor on a 100% trophy
    # would otherwise exceed it).
    ceilings = [min(1.0, max(0.0, c)) for c in ceilings]

    # Refresh volatility for the spread: the pooled RMS logit move
    # at real refreshes, clamped to [0.3, 0.8]; 0.5 where the
    # history has no refresh to calibrate from.
    vol = refresh_volatility(reads)
    spread_floor = SPREAD_FLOOR if vol is None else min(
        SPREAD_CAL_MAX, max(SPREAD_CAL_MIN, vol))

    # Earliness weight from the last rate, for the flat-history prior:
    # 1 for the highest-rate trophy, 0 for the lowest (100% excluded).
    lg = [logit(r) for r in last_rates]
    non100 = [v for r, v in zip(last_rates, lg) if r < 99.99]
    lo_lg = min(non100) if non100 else 0.0
    hi_lg = max(non100) if non100 else 0.0
    span = hi_lg - lo_lg if hi_lg > lo_lg else 1.0

    # Expected refreshes in the horizon (Poisson mean), for the prior.
    # The flat-history prior assumes a refresh happens; its size is set
    # by the player-growth factor below.
    medians = []
    sigmas = []
    all_flat = flat_players
    for i in range(n):
        # Earners series for this trophy.
        xs_e, ys_e = [], []
        for t, h in reads:
            rates = h.get("rates") or []
            r = rates[i] if i < len(rates) else None
            p = h.get("players_tracked")
            if r is not None and p and p > 0:
                e = r * p / 100.0
                xs_e.append(math.log1p(t))
                ys_e.append(math.log(e + 0.5))
        a_e, b_e, resid_e = linfit(xs_e, ys_e)
        flat_e = len(set(ys_e)) <= 1
        if not flat_e:
            all_flat = False
        b_e = min(max(b_e, 0.0), 3.0)
        # Damp the rate's part of the earners slope as the rate nears
        # its ceiling: growth is proportional to the headroom left.
        # The earners slope is (rate slope) + (players slope); only
        # the rate part saturates, and only when the rate is rising.
        # The projection pivots on the last read, not on the fit's
        # intercept, so damping the slope cannot pull it below where
        # the trophy already stands.
        b_r = b_e - b_p
        if b_r > 0:
            headroom = 1.0 - (last_rates[i] / 100.0) / ceilings[i]
            headroom = min(max(headroom, 0.0), 1.0)
            b_e = b_p + b_r * headroom
        e_last = last_rates[i] * last_players[i] / 100.0
        u_last = math.log1p(reads[-1][0])
        e_hat = e_last * math.exp(b_e * (u_target - u_last))
        e_hat = min(max(e_hat, e_last), p_hat)

        if flat_players and flat_e:
            # No signal in the history: apply the refresh prior.
            # A refresh adds NEW_PLAYERS_PER_REFRESH new players per
            # refresh in the horizon.  They have barely played, so
            # they hold this trophy at a fraction k of the old
            # players' rate; k grows with earliness (the current
            # rate, the best proxy for it) and averages ~0.07, as
            # the first refresh's totals imply.  The median is then
            # (old earners + new earners) / (old + new players).
            w = (lg[i] - lo_lg) / span if last_rates[i] < 99.99 else 1.0
            k = 0.02 + 0.12 * w
            n_ref = max(1.0, horizon_hours / REFRESH_HOURS)
            n_new = NEW_PLAYERS_PER_REFRESH * n_ref
            if last_rates[i] >= 99.99:
                # The intro trophy: every new player holds it too.
                med = 99.99
            else:
                e_new = n_new * (last_rates[i] / 100.0) * k
                med = 100.0 * (e_last + e_new) / (last_players[i] + n_new)
        else:
            med = 100.0 * e_hat / p_hat
        med = clamp(med)
        # Cap at the trophy's ceiling (see the ceilings block above).
        # This binds when the earners fit extrapolates explosive
        # launch growth.
        med = min(med, 100.0 * ceilings[i])
        med = clamp(med)
        medians.append(med)

        # Spread: random walk + refresh term + fit residual.  The
        # refresh term is the observed volatility where the history
        # has real refreshes, else the 0.5 floor.
        sigma_rw = SIGMA_RW * math.sqrt(horizon_days)
        sigma_fit = math.sqrt(resid_e) if resid_e > 0 else 0.0
        sigma = math.sqrt(sigma_rw ** 2 + spread_floor ** 2 + sigma_fit ** 2)
        sigmas.append(sigma)

    # If every series was flat, the fits above already used the prior
    # per trophy (flat_players and flat_e).  Nothing more to do here.
    _ = all_flat

    # ---- monotonicity the game forces ----
    # Platinum (id 1) <= every other trophy.
    if 1 in by_id and n > 1:
        plat = by_id[1]
        others = min(medians[j] for j in range(n) if j != plat)
        medians[plat] = min(medians[plat], others)
    # Story chain, in order: Gone Fishing, Chapter 1..4, final mission.
    # Identify them by description; fall back to ids 2..7 in order.
    chain = story_chain(trophies)
    if len(chain) >= 2:
        run = None
        for idx in chain:
            if run is None:
                run = medians[idx]
            else:
                run = min(run, medians[idx])
                medians[idx] = run
    # 100% Completion <= final mission.
    if 14 in by_id and 7 in by_id:
        medians[by_id[14]] = min(medians[by_id[14]], medians[by_id[7]])

    # ---- quantiles ----
    rows = []
    for i, trophy in enumerate(trophies):
        lg_med = logit(medians[i])
        s = sigmas[i]
        qs = []
        for z in Z:
            qs.append(round4(clamp(inv_logit(lg_med + z * s))))
        # Rounding can only tie; enforce non-decreasing defensively.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def story_chain(trophies):
    """Indices, in story order, of the trophies that form the main
    story chain: the intro mission, Chapter 1..4, the final mission.
    Matched by description; falls back to ids 2..7."""
    want = [
        ("gone fishing",),
        ("chapter 1",),
        ("chapter 2",),
        ("chapter 3",),
        ("chapter 4",),
        ("final mission",),
    ]
    idxs = []
    descs = [(t.get("description") or "").lower() for t in trophies]
    for keys in want:
        found = None
        for j, d in enumerate(descs):
            if all(k in d for k in keys):
                found = j
                break
        if found is None:
            return []
        idxs.append(found)
    # Must be in increasing id order to be a chain.
    ids = [trophies[j]["id"] for j in idxs]
    if ids != sorted(ids):
        return []
    return idxs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name} forecast over 65,536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
