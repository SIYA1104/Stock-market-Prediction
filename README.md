# 🧠 Beginner Stock Market Analysis (Learning Project)

This is a beginner-friendly **Stock Market Analysis Web Application** built using  
**Python, Streamlit, yfinance, pandas, plotly, and scikit-learn**.

The project allows you to:
- Visualize stock price history using candlestick charts  
- Generate SMA-based Buy/Sell signals  
- Predict next-day stock prices using machine learning models  
- Compare Linear Regression vs Random Forest  
- Download prediction results  

> ⚠️ **Note:** This project is for learning purposes only.  
> Not financial advice — do not use for real trading.

---

## 🚀 Features

### ✅ 1. Historical Price Visualization (Candlestick Chart)
- Fetches real-time stock data using Yahoo Finance API  
- Displays **Open, High, Low, Close** candlestick chart  
- Shows recent Close, Adj Close, and Volume values  

### ✅ 2. SMA Strategy (Buy/Sell Signals)
- Calculates:
  - **Short-Term SMA**
  - **Long-Term SMA**
- Detects:
  - 📈 **Buy Signal:** Short SMA crosses above Long SMA  
  - 📉 **Sell Signal:** Short SMA crosses below Long SMA  
- Plots SMA with price candles  
- Displays last 5 buy/sell signals  

### ✅ 3. Machine Learning Price Predictor
Predicts **next-day closing price** using ML models:

**Models Used:**
- **Linear Regression**  
- **RandomForestRegressor**

**Features Used:**
- `Close_lag1`  
- `Close_lag2`  
- `SMA_5`

**Outputs:**
- RMSE comparison  
- Actual vs Predicted graph  
- Feature importance table  
- Downloadable CSV file

### ✅ 4. RSI Momentum Analysis
- Calculates **RSI (Relative Strength Index)** with 14-period default
- Identifies:
  - 🔴 **Overbought Signal:** RSI > 70 (potential sell)
  - 🟢 **Oversold Signal:** RSI < 30 (potential buy)
- Interactive RSI chart with signal zones
- Current RSI status indicator
- Historical overbought/oversold events  

---

## 📦 Tech Stack

| Component       | Library/Tool        |
|-----------------|----------------------|
| UI Framework    | Streamlit            |
| Data Source     | yfinance             |
| Data Processing | pandas, numpy        |
| Visualization   | plotly               |
| ML Algorithms   | scikit-learn         |
| Python Version  | 3.11                 |

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SIYA1104/Stock-market-Prediction.git
cd Stock-market-Prediction
```

### 2. Create Virtual Environment
```bash
python -m venv env
```

### 3. Activate Environment
```bash
# Windows
env\Scripts\activate

# macOS/Linux
source env/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
streamlit run app.py
```

Open your browser and go to `http://localhost:8501`

