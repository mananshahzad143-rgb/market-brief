"""
Market brief backend — relationship and decision engine.

All statistics happen here, once per run, in pandas. The page only renders.

Beyond raw series it computes:
  · rolling correlations      when each mechanism is working
  · regression with R²        how much a driver actually explains
  · lead-lag scan             the real delay from oil to inflation expectations
  · regime classification     which of four states we are in
  · historical base rates     what every asset did in this state before
  · signal scorecard          hit rate AND sample size, always both
  · fragility index           how dry the forest is
  · analogue engine           distribution of what followed similar days
  · equal-risk weights        volatility-adjusted sizing across the watchlist

Sources: yfinance, US Treasury, CoinGecko. No FRED; it blocks cloud servers.
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


def ann_vol(s, w=90):
    r = np.log(s / s.shift(1)).dropna().tail(w)
    return round(float(r.std() * np.sqrt(252) * 100), 1)


def pct(s, n):
    return None if len(s) <= n else round(float(s.iloc[-1] / s.iloc[-1 - n] - 1) * 100, 1)


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def regress(y, x):
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 200:
        return None
    b, a = np.polyfit(df["x"], df["y"], 1)
    pred = a + b * df["x"]
    ss_tot = ((df["y"] - df["y"].mean()) ** 2).sum()
    r2 = 1 - ((df["y"] - pred) ** 2).sum() / ss_tot if ss_tot else 0
    return {"slope": round(float(b), 3), "r2": round(float(r2), 3), "n": int(len(df))}


def lead_lag(driver, target, max_lag=180, step=10):
    dr, tg = driver.pct_change(21), target.diff(21)
    best, curve = None, []
    for lag in range(0, max_lag + 1, step):
        c = pd.concat([dr.shift(lag).rename("d"), tg.rename("t")], axis=1).dropna()
        if len(c) < 300:
            continue
        r = float(c["d"].corr(c["t"]))
        curve.append({"lag": lag, "corr": round(r, 3)})
        if best is None or abs(r) > abs(best["corr"]):
            best = {"lag": lag, "corr": round(r, 3)}
    return {"best": best, "curve": curve} if best else None


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
                   "rsi_z": z(float(r.iloc[-1]), r), "vol": ann_vol(s),
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
            rows[str(state)] = {"median": round(float(grp["f"].median()) * 100, 1),
                                "hit": round(float((grp["f"] > 0).mean()) * 100),
                                "n": int(len(grp))}
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
                      "hit": round(float(wins.mean()) * 100),
                      "base": round(float(base.mean()) * 100),
                      "median": round(float(fired["f"].median()) * 100, 1),
                      "horizon": horizon,
                      "active": bool(cond.dropna().iloc[-1]) if len(cond.dropna()) else False})

    if "real10" in tr and "gold" in mk:
        r = tr["real10"].dropna()
        rz = (r - r.rolling(504).mean()) / r.rolling(504).std()
        add("Real yield deeply low", rz < -1.0, mk["gold"], 90, "up",
            "Real yield more than 1σ below its two-year mean.")
        add("Real yield stretched high", rz > 1.0, mk["gold"], 90, "down",
            "Real yield more than 1σ above its mean; bonds outcompeting gold.")
    if "curve" in tr and "spx" in mk:
        add("Yield curve inverted", tr["curve"] < 0, mk["spx"], 252, "down",
            "10y below 2y. Preceded every US recession for fifty years, often by a year or more.")
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
            "label": "Elevated" if sc > .66 else "Moderate" if sc > .4 else "Low"}


def analogues(feat, target, horizon=90, k=25):
    df = feat.dropna()
    tgt = target.reindex(df.index).ffill()
    fwd = tgt.shift(-horizon) / tgt - 1
    if len(df) < 900:
        return None
    norm = (df - df.mean()) / df.std()
    today, pool = norm.iloc[-1], norm.iloc[:-400]
    if len(pool) < 300:
        return None
    near = (((pool - today) ** 2).sum(axis=1) ** 0.5).nsmallest(k).index
    r = fwd.loc[near].dropna()
    if len(r) < 8:
        return None
    return {"horizon": horizon, "n": int(len(r)),
            "median": round(float(r.median()) * 100, 1),
            "p10": round(float(r.quantile(.1)) * 100, 1),
            "p90": round(float(r.quantile(.9)) * 100, 1),
            "hit": round(float((r > 0).mean()) * 100),
            "dates": [d.strftime("%b %Y") for d in near[:6]],
            "returns": [round(float(x) * 100, 1) for x in sorted(r)]}


def panel(pid, title, mech, an, a, bn, b, invert_a=False, note=""):
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 300:
        return None
    ra = np.log(df["a"] / df["a"].shift(1)) if (df["a"] > 0).all() else df["a"].diff()
    rb = np.log(df["b"] / df["b"].shift(1))
    corr = ra.rolling(120).corr(rb)
    w = df.resample("W").last().dropna()
    cw = corr.resample("W").last().reindex(w.index).ffill()
    reg = regress(rb.rolling(21).sum(), ra.rolling(21).sum())
    return {"id": pid, "title": title, "mechanism": mech, "note": note,
            "dates": [d.strftime("%Y-%m-%d") for d in w.index],
            "a": {"name": an, "values": [round(float(x), 4) for x in w["a"]], "invert": invert_a},
            "b": {"name": bn, "values": [round(float(x), 4) for x in w["b"]]},
            "corr": [None if pd.isna(x) else round(float(x), 3) for x in cw],
            "corr_now": round(float(corr.dropna().iloc[-1]), 2),
            "corr_60d": round(float(corr.tail(60).mean()), 2),
            "regression": reg, "years": round(len(df) / 252, 1)}


def main():
    out = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "errors": []}
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
    for k in ("nom10", "nom2", "breakeven", "curve"):
        if k in tr and len(tr[k].dropna()):
            h[k] = round(float(tr[k].dropna().iloc[-1]), 2)
    for k in ("gold", "oil", "dxy", "spx", "copper"):
        if k in mk and len(mk[k].dropna()):
            h[k] = round(float(mk[k].dropna().iloc[-1]), 2)
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
                      ("dxy", "Dollar index", ""), ("spx", "S&P 500", ""), ("tlt", "Long bonds TLT", "$")):
        if k in mk:
            row(lab, mk[k], u)
    out["table"] = rows

    P = []
    try:
        if "real10" in tr and "gold" in mk:
            P.append(panel("rg", "Real yield against gold",
                "Gold pays nothing, so its only competitor is an inflation-protected bond. When "
                "the real yield falls, holding gold costs less and gold rises. The yield line is "
                "inverted here so the two should track together.",
                "10y real yield (inverted)", tr["real10"], "Gold", mk["gold"], True,
                "The master relationship. Everything else is a branch of it."))
        if "oil" in mk and "breakeven" in tr:
            P.append(panel("ob", "Oil against inflation expectations",
                "Oil is an input to nearly every price. When it rises the market raises its "
                "inflation forecast, which is what breakeven measures. That forecast is what turns "
                "a central bank hawkish — the mechanism by which a war can push gold down.",
                "WTI crude", mk["oil"], "10y breakeven", tr["breakeven"], False,
                "Oil → inflation expectations → central bank → real yields → gold."))
        if "dxy" in mk and "gold" in mk:
            P.append(panel("dg", "The dollar against gold",
                "Commodities are priced in dollars worldwide. A stronger dollar makes the same "
                "ounce dearer for every foreign buyer, so demand falls and the dollar price softens.",
                "Dollar index (inverted)", mk["dxy"], "Gold", mk["gold"], True))
        if "real10" in tr and "spx" in mk:
            P.append(panel("rs", "Real yield against the risk ladder",
                "Every asset competes for the same savings and the real yield is the referee. High "
                "real yields make safety genuinely attractive, and money leaves the top of the "
                "ladder first — where equities sit, and crypto sits above them.",
                "10y real yield (inverted)", tr["real10"], "S&P 500", mk["spx"], True))
        if "curve" in tr and "spx" in mk:
            P.append(panel("cs", "The yield curve as early warning",
                "When the 10 year yields less than the 2 year, the bond market is pricing a "
                "slowdown. It has preceded every US recession for fifty years, usually by six to "
                "eighteen months — long enough to feel like a false alarm.",
                "Curve 10y−2y", tr["curve"], "S&P 500", mk["spx"]))
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
