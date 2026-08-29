# Market brief

A macro and crypto brief that fetches itself. Python collects and computes on a
schedule; the HTML page just renders the result on your phone.

## What it tracks

**Macro, from FRED.** 10y real yield (the master series), nominal 10y and 2y,
breakeven inflation, the yield curve, the broad dollar index, WTI crude, the Fed
balance sheet, reverse repo, and the Treasury General Account.

**Derived.** Net liquidity (balance sheet less reverse repo less the Treasury
account) with its 8 week change, and 2y minus fed funds as a free substitute for
FedWatch: below zero means the bond market is pricing cuts.

**Gold**, from futures, with PAX Gold as backup.

**Crypto**, from CoinGecko: BTC, ETH, SOL, ROSE. RSI with its own z-score, 50 and
200 EMAs, annualised volatility, and beta to Bitcoin.

**Sentiment and leverage.** Fear and Greed with a year of history. Funding and
open interest from Deribit, which unlike Binance is reachable from US servers.

**The analogue engine.** Finds the days since 2003 whose macro state most
resembles today, then reports the distribution of what gold did over the next 90
days: median, 10th to 90th percentile, and hit rate. Never a point forecast.

## Setup, about ten minutes

1. Create a **public** GitHub repo named `market-brief`.

2. Add these files:

   ```
   fetch_data.py
   requirements.txt
   market-brief.html
   .github/workflows/update.yml     <- rename update.yml and put it here
   ```

3. Open the **Actions** tab, enable workflows, then run **Update market data**
   manually. It takes about a minute and commits `data.json`.

4. Open `data.json` in the repo, click **Raw**, and copy that URL. It looks like:

   ```
   https://raw.githubusercontent.com/YOURNAME/market-brief/main/data.json
   ```

5. In `market-brief.html`, set `DATA_URL` at the top of the script to that URL.
   Commit.

6. In **Settings → Pages**, set the source to your main branch. Your page appears
   at `https://YOURNAME.github.io/market-brief/market-brief.html`.

7. On your iPhone, open that URL in Safari, then Share → Add to Home Screen.

From then on it updates twice a day on its own.

## Notes

The repo must be public for raw.githubusercontent.com to serve the file to your
browser. Nothing sensitive is stored in it: only public market data.

If `DATA_URL` is left blank the page still works by fetching live in the browser
through public relays. That path is less reliable, which is the reason this
backend exists.

Predictions you log are stored on your device, not in the repo.

## Limits worth keeping in mind

Crypto statistics rest on roughly one and a half cycles. The macro series have
twenty years. Do not let confidence earned on the second leak into the first.

The analogue engine's range matters more than its median. A wide range spanning
both directions means the setup carries no reliable edge, and that is a genuine
finding rather than a failure.

Fragility measures how dry the forest is, not when lightning strikes. It should
change position size, never generate an entry or exit.
