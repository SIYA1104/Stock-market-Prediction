# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Beginner Stock ML", layout="wide")

st.title("Beginner Stock Market Analysis (learning project)")
st.markdown("Not financial advice. Educational only — simple data + models for learning.")

# ---- Sidebar inputs ----
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL, MSFT, TCS.NS)", value="AAPL")
days = st.sidebar.number_input("Days historical", min_value=60, max_value=2000, value=365)
short_window = st.sidebar.number_input("Short SMA days", min_value=2, max_value=200, value=20)
long_window = st.sidebar.number_input("Long SMA days", min_value=2, max_value=400, value=50)

if st.sidebar.button("Refresh data"):
    # Use the supported rerun (newer Streamlit)
    st.rerun()

# ---- Robust fetch_data ----
@st.cache_data(show_spinner=False)
def fetch_data(ticker, days):
    """
    Robust data fetcher:
    1) Try yf.download
    2) Try yf.Ticker(...).history as fallback
    3) If still empty and ticker looks like an NSE symbol, try adding '.NS'
    Returns a cleaned dataframe with at least Open,High,Low,Close, or None if not available.
    """
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=int(days * 1.2))

    def clean_df(dfr):
        if dfr is None or dfr.empty:
            return None
        # flatten MultiIndex
        if isinstance(dfr.columns, pd.MultiIndex):
            dfr.columns = [' '.join(map(str, col)).strip() for col in dfr.columns.values]
        required = ['Open', 'High', 'Low', 'Close']
        if not all(r in dfr.columns for r in required):
            return None
        # keep available desired columns
        desired = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available = [c for c in desired if c in dfr.columns]
        dfr = dfr[available].copy()
        dfr.dropna(inplace=True)
        dfr.index = pd.to_datetime(dfr.index)
        dfr = dfr.tail(days)
        if dfr.empty:
            return None
        return dfr

    # 1) primary attempt: yf.download
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
    except Exception:
        df = None
    df = clean_df(df)
    if df is not None:
        return df

    # 2) fallback: Ticker.history()
    try:
        t = yf.Ticker(ticker)
        df2 = t.history(start=start, end=end, period=None)
    except Exception:
        df2 = None
    df2 = clean_df(df2)
    if df2 is not None:
        return df2

    # 3) If ticker likely NSE and doesn't already have .NS, try with .NS
    if not ticker.upper().endswith('.NS'):
        try_ticker = ticker + '.NS'
        try:
            df3 = yf.download(try_ticker, start=start, end=end, progress=False)
        except Exception:
            df3 = None
        df3 = clean_df(df3)
        if df3 is not None:
            return df3
        # try Ticker.history too
        try:
            t3 = yf.Ticker(try_ticker)
            df4 = t3.history(start=start, end=end)
        except Exception:
            df4 = None
        df4 = clean_df(df4)
        if df4 is not None:
            return df4

    # nothing worked
    return None

# ---- Fetch data ----
df = fetch_data(ticker, days)

# If you want to debug the raw yfinance response, uncomment the following temporary block:
# _raw_try = yf.download(ticker, start=datetime.datetime.today()-datetime.timedelta(days=int(days*1.2)),
#                        end=datetime.datetime.today(), progress=False)
# st.write("DEBUG: raw df columns:", list(_raw_try.columns))
# st.dataframe(_raw_try.head(5))

if df is None:
    st.error("No usable price data found for this ticker. Try a different symbol (e.g. AAPL, MSFT, TCS.NS).")
    st.stop()

# ---- Tabs for the three functions ----
tab1, tab2, tab3 = st.tabs(["Price Chart", "SMA & Signals", "Simple Predictor"])

with tab1:
    st.subheader(f"{ticker} — Price chart (last {len(df)} days)")
    # Candlestick (requires OHLC)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='OHLC'
    ))
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.write("Showing recent price stats:")
    display_cols = [c for c in ['Close', 'Adj Close', 'Volume'] if c in df.columns]
    st.dataframe(df[display_cols].tail(5))

with tab2:
    st.subheader("Simple Moving Averages & Crossover Signals")
    data = df.copy()
    data['SMA_short'] = data['Close'].rolling(short_window).mean()
    data['SMA_long'] = data['Close'].rolling(long_window).mean()
    data.dropna(inplace=True)
    if data.empty:
        st.warning("Not enough data to compute SMAs with the chosen windows.")
    else:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(width=1)))
        fig2.add_trace(go.Scatter(x=data.index, y=data['SMA_short'], name=f'SMA {short_window}'))
        fig2.add_trace(go.Scatter(x=data.index, y=data['SMA_long'], name=f'SMA {long_window}'))
        fig2.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        # signals
        data['signal'] = 0
        data.loc[data['SMA_short'] > data['SMA_long'], 'signal'] = 1
        data['signal_change'] = data['signal'].diff()
        buys = data[data['signal_change'] == 1]
        sells = data[data['signal_change'] == -1]
        st.write(f"Buy signals (cross above): {len(buys)} — showing last 5")
        if not buys.empty:
            st.dataframe(buys[['Close', 'SMA_short', 'SMA_long']].tail(5))
        else:
            st.write("No recent buy signals found.")
        st.write(f"Sell signals (cross below): {len(sells)} — showing last 5")
        if not sells.empty:
            st.dataframe(sells[['Close', 'SMA_short', 'SMA_long']].tail(5))
        else:
            st.write("No recent sell signals found.")

with tab3:
    st.subheader("Very basic next-day Close predictor (Linear Regression vs RandomForest)")
    # Ensure Close exists (fetch_data guarantees it, but double-check)
    if 'Close' not in df.columns:
        st.error("Not enough price data (no 'Close' column) to run predictor.")
    else:
        data = df.copy()
        data['Close_lag1'] = data['Close'].shift(1)
        data['Close_lag2'] = data['Close'].shift(2)
        data['SMA_5'] = data['Close'].rolling(5).mean().shift(1)
        data['Target'] = data['Close']
        data.dropna(inplace=True)

        if len(data) < 30:
            st.warning("Not enough data to train reliably. Increase 'Days historical'.")
        else:
            data['Target_next'] = data['Target'].shift(-1)
            data.dropna(inplace=True)

            FEATURES = ['Close_lag1', 'Close_lag2', 'SMA_5']
            FEATURES = [f for f in FEATURES if f in data.columns]
            X = data[FEATURES].values
            y = data['Target_next'].values

            # train-test split (time-series style)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            preds_lr = lr.predict(X_test)
            rmse_lr = np.sqrt(mean_squared_error(y_test, preds_lr))

            # RandomForest (simple, beginner-friendly)
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            preds_rf = rf.predict(X_test)
            rmse_rf = np.sqrt(mean_squared_error(y_test, preds_rf))

            st.write(f"LinearRegression RMSE: `{rmse_lr:.4f}` — RandomForest RMSE: `{rmse_rf:.4f}`")

            # predict next day using latest features
            last_row = data.iloc[-1]
            next_features = np.array([last_row[f] for f in FEATURES]).reshape(1, -1)
            next_pred_lr = lr.predict(next_features)[0]
            next_pred_rf = rf.predict(next_features)[0]
            st.success(f"Next-day prediction — LinearRegression: {next_pred_lr:.2f} | RandomForest: {next_pred_rf:.2f}")

            # Compare plot for test set
            compare_df = pd.DataFrame({
                "Actual": y_test.flatten(),
                "LinearPred": preds_lr.flatten(),
                "RF_Pred": preds_rf.flatten()
            }, index=pd.RangeIndex(start=0, stop=len(y_test)))

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=compare_df.index, y=compare_df['Actual'], mode='lines+markers', name='Actual'))
            fig_cmp.add_trace(go.Scatter(x=compare_df.index, y=compare_df['LinearPred'], mode='lines+markers', name='LinearPred'))
            fig_cmp.add_trace(go.Scatter(x=compare_df.index, y=compare_df['RF_Pred'], mode='lines+markers', name='RF_Pred'))
            fig_cmp.update_layout(title="Actual vs Predicted (test set)", height=360, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_cmp, use_container_width=True)

            # RandomForest feature importances
            if hasattr(rf, "feature_importances_"):
                fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
                st.write("RandomForest feature importances:")
                st.dataframe(fi.reset_index().rename(columns={'index':'feature', 0:'importance'}))

            # Download CSV of test-set results
            download_df = compare_df.copy()
            download_df['index'] = download_df.index
            csv = download_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download test-set results CSV", csv, file_name=f"{ticker}_predictions.csv", mime='text/csv')

# ---- Footer / tips ----
st.markdown("---")
st.write("Tips: Try different tickers, change SMA windows, or replace LinearRegression with other models (RandomForest, simple LSTM later).")
