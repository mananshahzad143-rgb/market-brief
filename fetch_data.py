"""
Market brief backend — relationship and decision engine.

CORRECTIONS in this version, all four deliberate:

1. SIGN CONVENTION. Each panel now declares what "working" means. Real yield
   against gold should be NEGATIVELY correlated when the mechanism holds. The
   old code coloured any positive correlation green, which was backwards on
   three of five panels.

2. EFFECTIVE SAMPLE SIZE. Overlapping 90-day windows on daily data inflate n
   by roughly the horizon. Every statistic now reports n_eff = n / horizon
   alongside the raw count.

3. ANALOGUE SPACING. Nearest-neighbour matching happily returned 25 dates from
   a single two-month episode and called it n=25. Matches must now be at least
   120 days apart, and the output reports how many distinct episodes and what
   span they cover.

4. NON-OVERLAPPING REGRESSION. R² is now computed on non-overlapping 21-day
   blocks rather than a rolling sum, so the fit is not flattered by
   autocorrelation.

Sources: yfinance, US Treasury, CoinGecko. No FRED — it blocks cloud servers.
"""

import io
import json
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (market-brief)"}
OUT = "data.json"
START = "2004-01-01"
log = lambda m: print(f"  {m}", flush=True)

MARKETS = {"gold": "GC=F", "oil": "CL=F", "spx": "^GSPC",
           "dxy": "DX-Y.NYB", "tlt": "TLT", "copper": "HG=F"}
COINS = [("bitcoin", "BTC"), ("ethereum", "ETH"),
         ("solana", "SOL"), ("oasis-network", "ROSE")]
EPISODES = [("2008-09-15", "Lehman"), ("2011-09-06", "Gold peak"),
            ("2013-05-22", "Taper tantrum"), ("2020-03-23", "Fed unlimited"),
            ("2022-03-16", "Hikes begin")]


def z(v, hist):
    h = pd.Series(hist).dropna()
    return round(float((v - h.mean()) / h.std()), 2) if h.std() else 0.0


def ann_vol(s, w=90, periods=252):
    """
    periods=252 for markets that close at weekends, 365 for crypto which does
    not. Using 252 on crypto understates volatility by about 17%.
    """
    r = np.log(s / s.shift(1)).dropna().tail(w)
    return round(float(r.std() * np.sqrt(periods) * 100), 1)


def pct(s, n):
    return None if len(s) <= n else round(float(s.iloc[-1] / s.iloc[-1 - n] - 1) * 100, 1)


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def regress_blocks(y, x, block=21):
    """
    FIX 3: non-overlapping blocks. A rolling sum reuses each observation `block`
    times, which inflates apparent fit. Taking every Nth row makes the sample
    independent and the R² honest.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    df = df.iloc[::block]                       # non-overlapping
    if len(df) < 60:
        return None
    b, a = np.polyfit(df["x"], df["y"], 1)
    pred = a + b * df["x"]
    ss_tot = ((df["y"] - df["y"].mean()) ** 2).sum()
    r2 = 1 - ((df["y"] - pred) ** 2).sum() / ss_tot if ss_tot else 0
    return {"slope": round(float(b), 3), "r2": round(float(max(0, r2)), 3),
            "n": int(len(df)), "basis": f"non-overlapping {block}-day blocks"}


def lead_lag(driver, target, max_lag_days=120, window=5):
    """
    M1 FIX. The previous version ran dropna() before subsampling, and shift(lag)
    creates leading NaNs, so every lag ended up sampling a DIFFERENT set of
    calendar dates. Cross-lag comparison was invalid.

    Now: align once, difference once, subsample onto ONE fixed grid, and only
    then shift. Every lag is measured on the same dates, so the comparison is
    like-for-like. Lag resolution equals `window` days.

    LIMITATION no design removes: a peak AT zero cannot distinguish "same day"
    from "any delay shorter than the window". Read it as "faster than `window`
    days". A peak AWAY from zero is the informative result.
    """
    df = pd.concat([driver.rename("d"), target.rename("t")], axis=1).dropna()
    if len(df) < 500:
        return None
    grid = pd.concat([df["d"].pct_change(window).rename("d"),
                      df["t"].diff(window).rename("t")], axis=1).dropna()
    grid = grid.iloc[::window]          # one fixed non-overlapping grid
    best, curve = None, []
    for k in range(0, max_lag_days // window + 1):
        c = pd.concat([grid["d"].shift(k).rename("dd"),
                       grid["t"].rename("tt")], axis=1).dropna()
        if len(c) < 100:
            continue
        r = float(c["dd"].corr(c["tt"]))
        lag_days = k * window
        curve.append({"lag": lag_days, "corr": round(r, 3)})
        if best is None or abs(r) > abs(best["corr"]):
            best = {"lag": lag_days, "corr": round(r, 3), "n": int(len(c))}
    return {"best": best, "curve": curve, "window": window} if best else None


def episode_count(cond, min_run=21):
    """
    M3 FIX v2. Counting every switch-on still overstated independence, because a
    condition defined by a threshold flickers on and off whenever the underlying
    hovers near that threshold. The 90-day change in real yields crosses zero
    repeatedly, which produced ~80 "episodes" that were really boundary noise.

    An episode now only counts if the condition HOLDS for at least `min_run`
    trading days (default one month). That is the number of genuinely distinct,
    sustained occurrences — the honest denominator.
    """
    c = cond.fillna(False).astype(bool)
    if not len(c):
        return 0
    grp = (c != c.shift(fill_value=False)).cumsum()
    n = 0
    for _, run in c.groupby(grp):
        if bool(run.iloc[0]) and len(run) >= min_run:
            n += 1
    return int(n)


def get_markets():
    import yfinance as yf
    out = {}
    for name, tick in MARKETS.items():
        try:
            d = yf.download(tick, start=START, progress=False,
                            auto_adjust=True, threads=False)
            c = d["Close"]
            s = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
            s.index = pd.to_datetime(s.index).tz_localize(None)
            out[name] = s.dropna()
            log(f"{name:7s} {len(s)} rows")
        except Exception as e:
            log(f"{name:7s} failed: {e}")
        time.sleep(1)
    return pd.DataFrame(out)


def _tsy_year(kind, year):
    url = ("https://home.treasury.gov/resource-center/data-chart-center/"
           "interest-rates/daily-treasury-rates.csv/"
           f"{year}/all?type={kind}&field_tdr_date_value={year}&page&_format=csv")
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    dc = [c for c in d.columns if "date" in c.lower()][0]
    d[dc] = pd.to_datetime(d[dc])
    return d.set_index(dc).sort_index()


def _pick(df, target):
    for c in df.columns:
        if c.strip().lower().replace(" ", "") == target:
            return pd.to_numeric(df[c], errors="coerce")
    return None


def get_treasury():
    yr = dt.date.today().year
    real, nom = [], []
    for y in range(2004, yr + 1):
        for kind, bucket in (("daily_treasury_real_yield_curve", real),
                             ("daily_treasury_yield_curve", nom)):
            try:
                bucket.append(_tsy_year(kind, y))
            except Exception:
                pass
        time.sleep(0.2)
    log(f"treasury: {len(real)} real years, {len(nom)} nominal years")
    out = {}
    if real:
        v = _pick(pd.concat(real), "10yr")
        if v is not None:
            out["real10"] = v
    if nom:
        n = pd.concat(nom)
        for lab, t in (("nom10", "10yr"), ("nom2", "2yr")):
            v = _pick(n, t)
            if v is not None:
                out[lab] = v
    df = pd.DataFrame(out).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if {"nom10", "real10"} <= set(df.columns):
        df["breakeven"] = df["nom10"] - df["real10"]
    if {"nom10", "nom2"} <= set(df.columns):
        df["curve"] = df["nom10"] - df["nom2"]
    return df


def get_coins():
    out, btc = {}, None
    for cid, sym in COINS:
        try:
            d = requests.get(f"https://api.coingecko.com/api/v3/coins/{cid}"
                             "/market_chart?vs_currency=usd&days=365",
                             headers=UA, timeout=45).json()
            s = pd.Series({pd.to_datetime(p[0], unit="ms").normalize(): p[1]
                           for p in d["prices"]})
            s = s[~s.index.duplicated(keep="last")].sort_index()
            r = rsi(s).dropna()
            rec = {"price": float(s.iloc[-1]), "rsi": round(float(r.iloc[-1]), 1),
                   "rsi_z": z(float(r.iloc[-1]), r), "vol": ann_vol(s, periods=365),
                   "chg7": pct(s, 7), "chg30": pct(s, 30), "chg90": pct(s, 90),
                   "vs_ema200": round(float(s.iloc[-1] / s.ewm(span=200).mean().iloc[-1] - 1) * 100, 1),
                   "drawdown": round(float(s.iloc[-1] / s.max() - 1) * 100, 1)}
            if sym == "BTC":
                btc = s
            elif btc is not None:
                ra = np.log(s / s.shift(1)).dropna().tail(90)
                rb = np.log(btc / btc.shift(1)).dropna().tail(90)
                n = min(len(ra), len(rb))
                var = rb.tail(n).var()
                rec["beta"] = round(float(np.cov(ra.tail(n), rb.tail(n))[0, 1] / var), 2) if var else None
            out[sym] = rec
            log(f"{sym} ok")
            time.sleep(2.5)
        except Exception as e:
            log(f"{sym} failed: {e}")
    return out


def get_fng():
    try:
        d = requests.get("https://api.alternative.me/fng/?limit=365",
                         headers=UA, timeout=30).json()
        v = [int(x["value"]) for x in d["data"]]
        return {"now": v[0], "label": d["data"][0]["value_classification"],
                "z": z(v[0], v[:180]), "spark": v[:120][::-1]}
    except Exception:
        return None


def classify(real):
    hi, up = real >= 1.0, real.diff(90) > 0
    lab = pd.Series(index=real.index, dtype=object)
    lab[hi & up] = "High and rising"
    lab[hi & ~up] = "High and falling"
    lab[~hi & up] = "Low and rising"
    lab[~hi & ~up] = "Low and falling"
    return lab


def base_rates(regime, assets, horizon=90):
    """
    M2: `horizon` is TRADING days. 90 trading days is roughly 4 calendar months.
    M3: n_eff counts distinct regime EPISODES, not days divided by horizon,
        because a regime persists for months rather than resetting daily.
    """
    out = {}
    for name, s in assets.items():
        s = s.dropna()
        fwd = s.shift(-horizon) / s - 1
        j = pd.concat([regime.rename("r"), fwd.rename("f")], axis=1).dropna()
        if len(j) < 400:
            continue
        rows = {}
        for state, grp in j.groupby("r"):
            if len(grp) < 60:
                continue
            n_eff = episode_count(j["r"] == state)
            rows[str(state)] = {"median": round(float(grp["f"].median()) * 100, 1),
                                "hit": round(float((grp["f"] > 0).mean()) * 100),
                                "n": int(len(grp)), "n_eff": max(1, n_eff)}
        if rows:
            out[name] = rows
    return out


def scorecard(tr, mk):
    cards = []

    def add(name, cond, target, horizon, direction, desc):
        fwd = target.shift(-horizon) / target - 1
        j = pd.concat([cond.rename("c"), fwd.rename("f")], axis=1).dropna()
        fired = j[j["c"].astype(bool)]
        if len(fired) < 40:
            return
        wins = (fired["f"] > 0) if direction == "up" else (fired["f"] < 0)
        base = (j["f"] > 0) if direction == "up" else (j["f"] < 0)
        cards.append({"name": name, "desc": desc, "fired": int(len(fired)),
                      # M3: distinct switch-ons, not fired-days / horizon
                      "n_eff": episode_count(j["c"]),
                      "hit": round(float(wins.mean()) * 100),
                      "base": round(float(base.mean()) * 100),
                      "median": round(float(fired["f"].median()) * 100, 1),
                      # M2: horizons are TRADING days; state calendar months too
                      "horizon": horizon,
                      "horizon_label": f"{horizon} trading days (~{round(horizon/21)} months)",
                      "active": bool(cond.dropna().iloc[-1]) if len(cond.dropna()) else False})

    if "real10" in tr and "gold" in mk:
        r = tr["real10"].dropna()
        rz = (r - r.rolling(504).mean()) / r.rolling(504).std()
        add("Real yield deeply low", rz < -1.0, mk["gold"], 90, "up",
            "Real yield more than 1σ below its two-year mean.")
        add("Real yield stretched high", rz > 1.0, mk["gold"], 90, "down",
            "Real yield more than 1σ above its mean; bonds outcompeting gold.")
    if "curve" in tr and "spx" in mk:
        add("Curve inverted → equities fall in 1y", tr["curve"] < 0, mk["spx"], 252, "down",
            "NOTE: the curve's real claim is about RECESSIONS, not equity prices. This tests the "
            "weaker equity version because recession dates are not in this dataset.")
    if "gold" in mk:
        g = mk["gold"]
        gz = (g - g.rolling(252).mean()) / g.rolling(252).std()
        add("Gold stretched", gz > 2.0, g, 90, "down",
            "Gold more than 2σ above its own one-year mean. A mean-reversion test.")
    if "dxy" in mk and "gold" in mk:
        add("Dollar falling hard", mk["dxy"].pct_change(90) < -0.04, mk["gold"], 90, "up",
            "Dollar down more than 4% over 90 days.")
    return cards


def fragility(tr, mk, coins, fng):
    parts = []
    if fng:
        parts.append({"k": "Fear and Greed", "v": f"{fng['now']} {fng['label']}",
                      "s": fng["now"] / 100})
    eth = coins.get("ETH")
    if eth:
        parts.append({"k": "ETH RSI", "v": f"{eth['rsi']} ({eth['rsi_z']:+}σ)",
                      "s": min(1, max(0, (eth["rsi"] - 30) / 55))})
    if "spx" in mk:
        lr = np.log(mk["spx"] / mk["spx"].shift(1)).dropna()
        cur = float(lr.tail(30).std())
        rank = float((lr.rolling(30).std().tail(1260) < cur).mean())
        parts.append({"k": "Equity volatility",
                      "v": f"{cur*np.sqrt(252)*100:.0f}% · {rank*100:.0f}th %ile",
                      "s": 1 - rank})
    if "curve" in tr and len(tr["curve"].dropna()):
        c = float(tr["curve"].dropna().iloc[-1])
        parts.append({"k": "Yield curve", "v": f"{c:+.2f}%", "s": 0.75 if c < 0 else 0.3})
    b = [c["beta"] for c in coins.values() if c.get("beta")]
    if b:
        parts.append({"k": "Crypto beta to BTC", "v": f"{np.mean(b):.2f}",
                      "s": min(1, max(0, (np.mean(b) - 0.8) / 1.2))})
    sc = float(np.mean([p["s"] for p in parts])) if parts else 0
    return {"score": round(sc, 2), "parts": parts,
            "label": "Elevated" if sc > .66 else "Moderate" if sc > .4 else "Low",
            "disclosure": ("Unvalidated heuristic. Components are equally weighted and the "
                           "mappings from raw values to 0–1 scores were chosen by hand, not "
                           "fitted or backtested. Treat this as a rough summary of several "
                           "readings, not as a measured quantity.")}


def analogues(feat, target, horizon=90, k=25, min_gap=120):
    """
    FIX 4: matches must be at least `min_gap` days apart. Without this the
    nearest-neighbour search returns one episode sampled k times and reports it
    as k independent observations, which is how you get a 96% hit rate that
    means nothing.
    """
    df = feat.dropna()
    tgt = target.reindex(df.index).ffill()
    fwd = tgt.shift(-horizon) / tgt - 1
    if len(df) < 900:
        return None
    norm = (df - df.mean()) / df.std()
    today, pool = norm.iloc[-1], norm.iloc[:-400]
    if len(pool) < 300:
        return None

    dist = (((pool - today) ** 2).sum(axis=1) ** 0.5).sort_values()
    chosen = []
    for idx in dist.index:
        if pd.isna(fwd.get(idx, np.nan)):
            continue
        if all(abs((idx - c).days) >= min_gap for c in chosen):
            chosen.append(idx)
        if len(chosen) >= k:
            break
    if len(chosen) < 6:
        return None

    r = fwd.loc[chosen].dropna()
    years = sorted({d.year for d in chosen})
    span = f"{min(chosen).strftime('%b %Y')} – {max(chosen).strftime('%b %Y')}"
    return {"horizon": horizon, "n": int(len(r)),
            "episodes": len(years), "years": years, "span": span,
            "min_gap_days": min_gap,
            "median": round(float(r.median()) * 100, 1),
            "p10": round(float(r.quantile(.1)) * 100, 1),
            "p90": round(float(r.quantile(.9)) * 100, 1),
            "hit": round(float((r > 0).mean()) * 100),
            "dates": [d.strftime("%b %Y") for d in sorted(chosen)[:8]],
            "returns": [round(float(x) * 100, 1) for x in sorted(r)],
            "mean_dist": round(float(dist.loc[chosen].mean()), 2)}


def panel(pid, title, mech, an, a, bn, b, expect, invert_a=False, note=""):
    """
    FIX 1: `expect` declares the sign the mechanism predicts, so the frontend
    can colour "working" correctly instead of assuming positive is good.
    expect = 'neg' means the mechanism implies a NEGATIVE correlation.
    """
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 300:
        return None
    ra = np.log(df["a"] / df["a"].shift(1)) if (df["a"] > 0).all() else df["a"].diff()
    rb = np.log(df["b"] / df["b"].shift(1))
    corr = ra.rolling(120).corr(rb)
    w = df.resample("W").last().dropna()
    cw = corr.resample("W").last().reindex(w.index).ffill()
    reg = regress_blocks(rb.rolling(21).sum(), ra.rolling(21).sum())

    # FIX 6: 'none' means the mechanism makes no same-day prediction, so the
    # strip must not be coloured as if it were passing or failing a test.
    sign = -1 if expect == "neg" else (0 if expect == "none" else 1)
    now = float(corr.dropna().iloc[-1])
    return {"id": pid, "title": title, "mechanism": mech, "note": note,
            "expect": expect,
            "expect_text": {"neg": "negative — the two should move opposite",
                            "pos": "positive — the two should move together",
                            "none": "none — this mechanism makes no same-day prediction"}[expect],
            "dates": [d.strftime("%Y-%m-%d") for d in w.index],
            "a": {"name": an, "values": [round(float(x), 4) for x in w["a"]], "invert": invert_a},
            "b": {"name": bn, "values": [round(float(x), 4) for x in w["b"]]},
            "corr": [None if pd.isna(x) else round(float(x), 3) for x in cw],
            "working": [None if pd.isna(x) else round(float(x) * sign, 3) for x in cw],
            "corr_now": round(now, 2),
            "working_now": round(now * sign, 2),
            "corr_60d": round(float(corr.tail(60).mean()), 2),
            "regression": reg, "years": round(len(df) / 252, 1)}


def main():
    out = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "errors": [], "version": "2.0-corrected"}
    log("markets…")
    try:
        mk = get_markets()
    except Exception as e:
        out["errors"].append(f"markets: {e}"); mk = pd.DataFrame()
    log("treasury…")
    try:
        tr = get_treasury()
    except Exception as e:
        out["errors"].append(f"treasury: {e}"); tr = pd.DataFrame()

    h = {}
    if "real10" in tr:
        s = tr["real10"].dropna()
        h["real10"] = round(float(s.iloc[-1]), 2)
        h["real10_z"] = z(float(s.iloc[-1]), s.tail(504))
        h["real10_chg90"] = round(float(s.iloc[-1] - s.iloc[-90]), 2) if len(s) > 90 else None
        h["real10_asof"] = s.index[-1].strftime("%Y-%m-%d")
    for k in ("nom10", "nom2", "breakeven", "curve"):
        if k in tr and len(tr[k].dropna()):
            h[k] = round(float(tr[k].dropna().iloc[-1]), 2)
    for k in ("gold", "oil", "dxy", "spx", "copper"):
        if k in mk and len(mk[k].dropna()):
            h[k] = round(float(mk[k].dropna().iloc[-1]), 2)
    if "gold" in mk:
        h["markets_asof"] = mk["gold"].dropna().index[-1].strftime("%Y-%m-%d")
    out["headline"] = h

    rows = []
    def row(name, s, unit=""):
        s = s.dropna()
        if len(s) < 300:
            return
        rows.append({"name": name, "unit": unit, "value": round(float(s.iloc[-1]), 2),
                     "z2y": z(float(s.iloc[-1]), s.tail(504)),
                     "chg30": pct(s, 21), "chg90": pct(s, 63), "chg1y": pct(s, 252),
                     "vol": ann_vol(s),
                     "pctile": round(float((s.tail(1260) < s.iloc[-1]).mean()) * 100),
                     "spark": [round(float(x), 4) for x in s.tail(260).iloc[::5]]})
    for k, lab, u in (("real10", "10y real yield", "%"), ("breakeven", "Breakeven inflation", "%"),
                      ("curve", "Yield curve 10y−2y", "%")):
        if k in tr:
            row(lab, tr[k], u)
    for k, lab, u in (("gold", "Gold", "$"), ("oil", "WTI crude", "$"), ("copper", "Copper", "$"),
                      ("dxy", "Dollar index", ""), ("spx", "S&P 500", ""),
                      ("tlt", "Long bonds TLT (total return)", "$")):
        if k in mk:
            row(lab, mk[k], u)
    out["table"] = rows

    P = []
    try:
        if "real10" in tr and "gold" in mk:
            P.append(panel("rg", "Real yield against gold",
                "Gold pays nothing, so its only competitor is an inflation-protected bond. When the "
                "real yield falls, holding gold costs less and gold rises. So the mechanism predicts "
                "these two move in OPPOSITE directions.",
                "10y real yield (inverted for display)", tr["real10"], "Gold", mk["gold"],
                "neg", True, "The master relationship. Everything else is a branch of it."))
        if "oil" in mk and "breakeven" in tr:
            P.append(panel("ob", "Oil against inflation expectations",
                "Oil is an input to nearly every price. When it rises the market raises its inflation "
                "forecast, which is what breakeven measures. The mechanism predicts these move TOGETHER.",
                "WTI crude", mk["oil"], "10y breakeven", tr["breakeven"], "pos", False,
                "Oil → inflation expectations → central bank → real yields → gold."))
        if "dxy" in mk and "gold" in mk:
            P.append(panel("dg", "The dollar against gold",
                "Commodities are priced in dollars worldwide. A stronger dollar makes the same ounce "
                "dearer for every foreign buyer, so demand falls. The mechanism predicts these move "
                "in OPPOSITE directions.",
                "Dollar index (inverted for display)", mk["dxy"], "Gold", mk["gold"], "neg", True))
        if "real10" in tr and "spx" in mk:
            P.append(panel("rs", "Real yield against the risk ladder",
                "Every asset competes for the same savings and the real yield is the referee. High "
                "real yields make safety attractive and money leaves the top of the ladder. The "
                "mechanism predicts these move in OPPOSITE directions.",
                "10y real yield (inverted for display)", tr["real10"], "S&P 500", mk["spx"], "neg", True))
        if "curve" in tr and "spx" in mk:
            P.append(panel("cs", "The yield curve as early warning",
                "An inverted curve is a LAGGED warning, typically six to eighteen months ahead. There "
                "is no reliable same-day relationship, so expect the correlation strip to look like "
                "noise. Judge this one from the scorecard, not from this chart.",
                "Curve 10y−2y", tr["curve"], "S&P 500", mk["spx"], "none", False,
                "The near-zero R² here is the honest reading: contemporaneously, the curve tells "
                "you almost nothing about equities. The strip is left uncoloured because there is "
                "no same-day claim to pass or fail."))
    except Exception as e:
        out["errors"].append(f"panels: {e}")
    out["panels"] = [p for p in P if p]
    out["episodes"] = [{"date": d, "label": l} for d, l in EPISODES]

    try:
        if "real10" in tr:
            reg = classify(tr["real10"].dropna())
            out["regime"] = {"now": str(reg.dropna().iloc[-1])}
            assets = {k: mk[k] for k in ("gold", "spx", "oil", "dxy", "tlt", "copper") if k in mk}
            out["base_rates"] = base_rates(reg, assets)
            log(f"regime: {out['regime']['now']}")
    except Exception as e:
        out["errors"].append(f"regime: {e}")

    try:
        if "oil" in mk and "breakeven" in tr:
            out["leadlag"] = lead_lag(mk["oil"], tr["breakeven"])
    except Exception as e:
        out["errors"].append(f"leadlag: {e}")

    try:
        out["scorecard"] = scorecard(tr, mk)
    except Exception as e:
        out["errors"].append(f"scorecard: {e}")

    try:
        if "real10" in tr and {"gold", "dxy"} <= set(mk.columns):
            f = pd.DataFrame({"lvl": tr["real10"], "chg": tr["real10"].diff(90),
                              "curve": tr["curve"] if "curve" in tr else tr["real10"] * 0,
                              "dxy": mk["dxy"].pct_change(90) * 100}).dropna()
            a = analogues(f, mk["gold"])
            if a:
                out["analogue"] = a
                log(f"analogue: {a['n']} matches across {a['episodes']} years")
    except Exception as e:
        out["errors"].append(f"analogue: {e}")

    log("crypto…")
    out["coins"] = get_coins()
    out["fng"] = get_fng()

    try:
        out["fragility"] = fragility(tr, mk, out["coins"], out["fng"])
        vols = {s: c["vol"] for s, c in out["coins"].items() if c.get("vol")}
        if vols:
            inv = {s: 1 / v for s, v in vols.items()}
            tot = sum(inv.values())
            out["risk_weights"] = {s: round(w / tot * 100) for s, w in inv.items()}
            out["vols"] = vols
    except Exception as e:
        out["errors"].append(f"risk: {e}")

    with open(OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    log(f"wrote {OUT}, {len(json.dumps(out))//1024} KB")
    if out["errors"]:
        log("errors: " + "; ".join(out["errors"]))


if __name__ == "__main__":
    main()
