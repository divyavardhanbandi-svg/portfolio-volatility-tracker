"""
Portfolio Volatility Tracker
----------------------------
Tracks the risk of a stock portfolio: rolling volatility, correlation,
Value at Risk (VaR), max drawdown, Sharpe ratio and volatility alerts.

Usage examples:
    python portfolio_volatility_tracker.py
    python portfolio_volatility_tracker.py --tickers AAPL MSFT GOOGL JPM --weights 0.3 0.3 0.2 0.2
    python portfolio_volatility_tracker.py --period 2y --window 21 --alert 0.25
    python portfolio_volatility_tracker.py --demo        # synthetic data, no internet needed
"""

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save charts to files, no display needed
import matplotlib.pyplot as plt

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------
def load_prices(tickers, period):
    """Download adjusted close prices from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed. Run: pip install yfinance")

    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if data.empty:
        sys.exit("No data downloaded. Check tickers and internet connection.")

    prices = data["Close"]
    if isinstance(prices, pd.Series):  # single ticker case
        prices = prices.to_frame(tickers[0])
    return prices.dropna(how="all").ffill().dropna()


def synthetic_prices(tickers, days=504, seed=42):
    """Generate random-walk prices for testing without internet."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    n = len(tickers)

    # Random correlated returns
    base = rng.normal(size=(n, n))
    cov = base @ base.T
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    vols = rng.uniform(0.15, 0.40, size=n) / np.sqrt(TRADING_DAYS)
    cov_daily = corr * np.outer(vols, vols)

    returns = rng.multivariate_normal(np.full(n, 0.0004), cov_daily, size=days)
    # Add a volatility shock in the last 30 days so alerts can trigger
    returns[-30:] *= 2.0
    prices = 100 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ----------------------------------------------------------------------
# 2. Risk calculations
# ----------------------------------------------------------------------
def compute_returns(prices):
    return prices.pct_change().dropna()


def portfolio_returns(returns, weights):
    return returns.dot(weights)


def annualized_volatility(returns):
    return returns.std() * np.sqrt(TRADING_DAYS)


def portfolio_volatility(returns, weights):
    """Volatility from the covariance matrix: sqrt(w' * Cov * w)."""
    cov = returns.cov() * TRADING_DAYS
    return float(np.sqrt(weights @ cov.values @ weights))


def rolling_volatility(series, window):
    return series.rolling(window).std() * np.sqrt(TRADING_DAYS)


def value_at_risk(port_ret, confidence=0.95):
    """Historical 1-day VaR (returned as a positive loss percentage)."""
    return float(-np.percentile(port_ret, (1 - confidence) * 100))


def max_drawdown(port_ret):
    cumulative = (1 + port_ret).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    return float(drawdown.min()), drawdown


def sharpe_ratio(port_ret, risk_free=0.04):
    excess = port_ret.mean() * TRADING_DAYS - risk_free
    vol = port_ret.std() * np.sqrt(TRADING_DAYS)
    return float(excess / vol) if vol else float("nan")


def risk_contribution(returns, weights):
    """How much each asset contributes to total portfolio volatility."""
    cov = returns.cov() * TRADING_DAYS
    port_vol = np.sqrt(weights @ cov.values @ weights)
    marginal = cov.values @ weights / port_vol
    contrib = weights * marginal
    return pd.Series(contrib / contrib.sum(), index=returns.columns)


# ----------------------------------------------------------------------
# 3. Charts
# ----------------------------------------------------------------------
def make_charts(returns, port_ret, rolling_vol, drawdown, window, alert, out_prefix):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Rolling volatility with alert line
    ax = axes[0, 0]
    rolling_vol.plot(ax=ax, color="tab:blue")
    ax.axhline(alert, color="red", linestyle="--", label=f"Alert ({alert:.0%})")
    ax.set_title(f"{window}-Day Rolling Portfolio Volatility (annualized)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend()

    # Cumulative return
    ax = axes[0, 1]
    ((1 + port_ret).cumprod() - 1).plot(ax=ax, color="tab:green")
    ax.set_title("Cumulative Portfolio Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Drawdown
    ax = axes[1, 0]
    ax.fill_between(drawdown.index, drawdown.values, 0, color="tab:red", alpha=0.4)
    ax.set_title("Portfolio Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Correlation heatmap
    ax = axes[1, 1]
    corr = returns.corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Asset Correlation")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    path = f"{out_prefix}_dashboard.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Portfolio Volatility Tracker")
    p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "JPM", "XOM", "JNJ"])
    p.add_argument("--weights", nargs="+", type=float, help="Weights; default is equal weight")
    p.add_argument("--period", default="1y", help="History length, e.g. 6mo, 1y, 2y, 5y")
    p.add_argument("--window", type=int, default=30, help="Rolling window in trading days")
    p.add_argument("--alert", type=float, default=0.25, help="Annualized volatility alert level")
    p.add_argument("--confidence", type=float, default=0.95, help="VaR confidence level")
    p.add_argument("--out", default="portfolio", help="Output file prefix")
    p.add_argument("--demo", action="store_true", help="Use synthetic data (no internet)")
    args = p.parse_args()

    tickers = [t.upper() for t in args.tickers]

    # Weights
    if args.weights:
        if len(args.weights) != len(tickers):
            sys.exit("Number of weights must match number of tickers.")
        weights = np.array(args.weights, dtype=float)
        if weights.sum() <= 0:
            sys.exit("Weights must sum to a positive number.")
        weights = weights / weights.sum()  # normalize to 1
    else:
        weights = np.full(len(tickers), 1 / len(tickers))

    # Data
    prices = synthetic_prices(tickers) if args.demo else load_prices(tickers, args.period)
    tickers = list(prices.columns)  # in case a ticker failed to download
    if len(tickers) != len(weights):
        sys.exit("Some tickers returned no data; please fix the ticker list or weights.")

    returns = compute_returns(prices)
    port_ret = portfolio_returns(returns, weights)
    roll_vol = rolling_volatility(port_ret, args.window).dropna()
    mdd, drawdown = max_drawdown(port_ret)

    # ---- Report ----
    print("=" * 60)
    print(" PORTFOLIO VOLATILITY REPORT")
    print("=" * 60)
    print(f" Period: {prices.index[0].date()} to {prices.index[-1].date()}  ({len(returns)} days)")
    print("\n Holdings:")
    for t, w in zip(tickers, weights):
        print(f"   {t:<8} {w:>6.1%}")

    print("\n Individual annualized volatility:")
    for t, v in annualized_volatility(returns).items():
        print(f"   {t:<8} {v:>6.1%}")

    port_vol = portfolio_volatility(returns, weights)
    avg_vol = float((annualized_volatility(returns) * weights).sum())
    print(f"\n Portfolio volatility:        {port_vol:.1%}")
    print(f" Diversification benefit:     {avg_vol - port_vol:.1%} lower than weighted average ({avg_vol:.1%})")
    print(f" Annualized return:           {port_ret.mean() * TRADING_DAYS:.1%}")
    print(f" Sharpe ratio:                {sharpe_ratio(port_ret):.2f}")
    print(f" 1-day VaR ({args.confidence:.0%}):          {value_at_risk(port_ret, args.confidence):.2%}")
    print(f" Max drawdown:                {mdd:.1%}")

    print("\n Share of portfolio risk by asset:")
    for t, c in risk_contribution(returns, weights).items():
        print(f"   {t:<8} {c:>6.1%}")

    # ---- Alert ----
    latest = float(roll_vol.iloc[-1])
    print("\n" + "-" * 60)
    if latest > args.alert:
        print(f" ALERT: {args.window}-day volatility is {latest:.1%}, above the {args.alert:.0%} threshold.")
    else:
        print(f" OK: {args.window}-day volatility is {latest:.1%}, below the {args.alert:.0%} threshold.")
    print("-" * 60)

    # ---- Save outputs ----
    out = pd.DataFrame({"portfolio_return": port_ret, "rolling_volatility": roll_vol, "drawdown": drawdown})
    csv_path = f"{args.out}_volatility.csv"
    out.to_csv(csv_path)
    png_path = make_charts(returns, port_ret, roll_vol, drawdown, args.window, args.alert, args.out)
    print(f"\n Saved: {csv_path}")
    print(f" Saved: {png_path}")


if __name__ == "__main__":
    main()
