"""
Market brief backend.

Runs on GitHub Actions. Fetches every source server-side, where CORS and
geo-blocks do not apply, computes all metrics, and writes data.json.
The phone just renders that file.

Every source is wrapped so one failure never kills the run.
"""

import io
import json
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (market-brief)"}
OUT = "data.json"

FRED_IDS = [
    "DFII10",      # 10y real yield  <- the master series
    "DGS10",       # 10y nominal
    "DGS2",        # 2y nominal
    "T10YIE",      # breakeven inflation
    "T10Y2Y",      # curve
    "DTWEXBGS",    # broad dollar
    "DCOILWTICO",  # WTI crude
    "DFF",         # effective fed funds
    "WALCL",       # Fed balance sheet (weekly)
    "RRPONTSYD",   # reverse repo
    "WTREGEN",     # Treasury general account
]

ASSETS = [
    ("bitcoin", "BTC", "Bitcoin"),
    ("ethereum", "ETH", "Ethereum"),
    ("solana", "SOL", "Solana"),
    ("oasis-network", "ROSE", "Oasis"),
]

log = lambda m: print(f"  {m}", flush=True)


# ────────────────────────────── maths ──────────────────────────────

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def zscore(value: float, hist: pd.Series) -> float:
    hist = hist.dropna()
    s = hist.std()
    return float((value - hist.mean()) / s) if s else 0.0


def ann_vol(closes: pd.Series, window: int = 90) -> float:
    r = np.log(closes / closes.shift(1)).dropna().tail(window)
    return float(r.std() * np.sqrt(365) * 100)


def beta_to(a: pd.Series, b: pd.Series, window: int = 90) -> float:
    ra = np.log(a / a.shift(1)).dropna().tail(window)
    rb = np.log(b / b.shift(1)).dropna().tail(window)
    n = min(len(ra), len(rb))
    ra, rb = ra.tail(n).values, rb.tail(n).values
    var = rb.var()
    return float(np.cov(ra, rb)[0, 1] / var) if var else 0.0


# ────────────────────────────── sources ──────────────────────────────

def get_fred() -> pd.DataFrame:
    """Fetch each series separately with retries. Small requests beat one large one."""
    frames = {}
    for sid in FRED_IDS:
        for attempt in range(3):
            try:
                url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
                       f"?id={sid}&cosd=2003-01-01")
                r = requests.get(url, headers=UA, timeout=12)
                r.raise_for_status()
                d = pd.read_csv(io.StringIO(r.text))
                d.columns = ["date", sid]
                d["date"] = pd.to_datetime(d["date"])
                frames[sid] = pd.to_numeric(d.set_index("date")[sid], errors="coerce")
                log(f"  {sid} ok")
                break
            except Exception as e:
                if attempt == 2:
                    log(f"  {sid} failed after 3 tries: {e}")
                time.sleep(1)
    if not frames:
        raise RuntimeError("no FRED series retrieved")
    return pd.DataFrame(frames)


def get_gold() -> pd.Series:
    """Gold futures. Falls back to PAX Gold, redeemable 1:1 for an ounce."""
    try:
        import yfinance as yf
        d = yf.download("GC=F", start="2003-01-01", progress=False, auto_adjust=True)
        if not d.empty:
            c = d["Close"]
            s = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s.dropna()
    except Exception as e:
        log(f"yfinance gold failed: {e}")
    d = requests.get(
        "https://api.coingecko.com/api/v3/coins/pax-gold/market_chart"
        "?vs_currency=usd&days=365", headers=UA, timeout=45).json()
    s = pd.Series({pd.to_datetime(p[0], unit="ms").normalize(): p[1] for p in d["prices"]})
    return s.sort_index()


def get_coin(cg_id: str) -> pd.Series:
    d = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
        f"?vs_currency=usd&days=365", headers=UA, timeout=45).json()
    s = pd.Series({pd.to_datetime(p[0], unit="ms").normalize(): p[1] for p in d["prices"]})
    return s[~s.index.duplicated(keep="last")].sort_index()


def get_fng() -> dict | None:
    d = requests.get("https://api.alternative.me/fng/?limit=365", headers=UA, timeout=30).json()
    vals = [int(x["value"]) for x in d["data"]]
    return {
        "now": vals[0],
        "label": d["data"][0]["value_classification"],
        "z": zscore(vals[0], pd.Series(vals[:180])),
        "hist": vals[:90][::-1],
    }


def get_derivs() -> dict:
    """Deribit is reachable from US runners; Binance is not."""
    out = {}
    for cur in ("BTC", "ETH"):
        try:
            d = requests.get(
                "https://www.deribit.com/api/v2/public/get_book_summary_by_instrument"
                f"?instrument_name={cur}-PERPETUAL", headers=UA, timeout=30).json()
            row = d["result"][0]
            out[cur] = {
                "funding_8h_pct": round(float(row.get("funding_8h", 0)) * 100, 5),
                "open_interest": float(row.get("open_interest", 0)),
                "mark": float(row.get("mark_price", 0)),
            }
        except Exception as e:
            log(f"deribit {cur} failed: {e}")
    return out


# ─────────────────────── the analogue engine ───────────────────────

def analogues(fred: pd.DataFrame, gold: pd.Series, horizon: int = 90, k: int = 25) -> dict | None:
    """
    Find the days in history whose macro state most resembles today, then
    report the DISTRIBUTION of what gold did next. Never a point forecast.
    """
    df = pd.DataFrame({
        "real": fred["DFII10"],
        "curve": fred["T10Y2Y"],
        "dxy": fred["DTWEXBGS"],
    }).join(gold.rename("gold"), how="inner").dropna()

    if len(df) < 800:
        return None

    feat = pd.DataFrame({
        "real_lvl": df["real"],
        "real_chg": df["real"].diff(90),
        "curve": df["curve"],
        "dxy_chg": df["dxy"].pct_change(90) * 100,
    }).dropna()

    df = df.loc[feat.index]
    fwd = df["gold"].shift(-horizon) / df["gold"] - 1

    norm = (feat - feat.mean()) / feat.std()
    today = norm.iloc[-1]

    # exclude the trailing year so matches don't overlap today's own window
    pool = norm.iloc[:-365]
    if len(pool) < 200:
        return None

    dist = ((pool - today) ** 2).sum(axis=1) ** 0.5
    near = dist.nsmallest(k).index
    rets = fwd.loc[near].dropna()
    if len(rets) < 8:
        return None

    return {
        "horizon_days": horizon,
        "n": int(len(rets)),
        "median_pct": round(float(rets.median()) * 100, 1),
        "p10_pct": round(float(rets.quantile(0.10)) * 100, 1),
        "p90_pct": round(float(rets.quantile(0.90)) * 100, 1),
        "hit_rate_pct": round(float((rets > 0).mean()) * 100, 0),
        "closest_dates": [d.strftime("%b %Y") for d in near[:5]],
        "features": {c: round(float(today[c]), 2) for c in norm.columns},
    }


# ────────────────────────────── assemble ──────────────────────────────

def main():
    out = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "errors": [],
    }

    # macro
    fred = None
    try:
        log("fetching FRED…")
        fred = get_fred()
        latest = {c: fred[c].dropna() for c in fred.columns}
        m = {}
        for c, s in latest.items():
            if len(s):
                m[c] = round(float(s.iloc[-1]), 4)
        # z-score and 90 day change on the master series
        rs = latest["DFII10"]
        m["DFII10_z"] = round(zscore(float(rs.iloc[-1]), rs.tail(504)), 2)
        m["DFII10_chg90"] = round(float(rs.iloc[-1] - rs.iloc[-90]), 2) if len(rs) > 90 else None

        # net liquidity in $tn, and its 8 week change
        w, rr, tg = latest["WALCL"], latest["RRPONTSYD"], latest["WTREGEN"]
        nl = pd.concat([w.rename("w"), rr.rename("r"), tg.rename("t")], axis=1).ffill().dropna()
        nl["net"] = (nl["w"] - nl["r"] - nl["t"]) / 1000
        m["net_liquidity_tn"] = round(float(nl["net"].iloc[-1]), 3)
        prior = nl["net"].iloc[-40] if len(nl) > 40 else nl["net"].iloc[0]
        m["net_liquidity_chg8w"] = round(float(nl["net"].iloc[-1] - prior), 3)

        # rate expectations, the free substitute for FedWatch
        m["rate_expectation"] = round(m["DGS2"] - m["DFF"], 2)
        out["macro"] = m
        log(f"real yield {m['DFII10']:+.2f}%")
    except Exception as e:
        out["errors"].append(f"fred: {e}")
        log(f"FRED failed: {e}")

    # gold
    gold = None
    try:
        log("fetching gold…")
        gold = get_gold()
        out["gold"] = {
            "price": round(float(gold.iloc[-1]), 2),
            "chg30_pct": round(float(gold.iloc[-1] / gold.iloc[-22] - 1) * 100, 1),
            "chg90_pct": round(float(gold.iloc[-1] / gold.iloc[-64] - 1) * 100, 1),
            "spark": [round(float(x), 2) for x in gold.tail(90)],
        }
    except Exception as e:
        out["errors"].append(f"gold: {e}")

    # crypto
    coins, btc = {}, None
    for cg_id, sym, name in ASSETS:
        try:
            log(f"fetching {sym}…")
            s = get_coin(cg_id)
            r = rsi(s).dropna()
            rec = {
                "name": name,
                "price": float(s.iloc[-1]),
                "rsi": round(float(r.iloc[-1]), 1),
                "rsi_z": round(zscore(float(r.iloc[-1]), r), 2),
                "ema200": round(float(s.ewm(span=200).mean().iloc[-1]), 6),
                "ema50": round(float(s.ewm(span=50).mean().iloc[-1]), 6),
                "vol_pct": round(ann_vol(s), 1),
                "chg7_pct": round(float(s.iloc[-1] / s.iloc[-8] - 1) * 100, 1),
                "chg30_pct": round(float(s.iloc[-1] / s.iloc[-31] - 1) * 100, 1),
                "spark": [round(float(x), 8) for x in s.tail(90)],
            }
            rec["vs_ema200_pct"] = round((rec["price"] / rec["ema200"] - 1) * 100, 1)
            coins[sym] = rec
            if sym == "BTC":
                btc = s
            else:
                coins[sym]["beta_btc"] = round(beta_to(s, btc), 2) if btc is not None else None
            time.sleep(2.5)  # stay inside the free rate limit
        except Exception as e:
            out["errors"].append(f"{sym}: {e}")
            log(f"{sym} failed: {e}")
    out["coins"] = coins

    # sentiment and derivatives
    try:
        out["fng"] = get_fng()
    except Exception as e:
        out["errors"].append(f"fng: {e}")
    out["derivs"] = get_derivs()

    # analogue engine
    if fred is not None and gold is not None:
        try:
            log("running analogue engine…")
            a = analogues(fred, gold)
            if a:
                out["analogue"] = a
                log(f"{a['n']} matches, median {a['median_pct']:+.1f}%")
        except Exception as e:
            out["errors"].append(f"analogue: {e}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    log(f"wrote {OUT} ({len(json.dumps(out))} bytes)")
    if out["errors"]:
        log("non-fatal errors: " + "; ".join(out["errors"]))


if __name__ == "__main__":
    main()
