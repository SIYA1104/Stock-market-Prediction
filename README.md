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
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
## 2. Create Virtual Environment
python -m venv env
## 3. Activate Environment
env\Scripts\activate.bat
## 4. Install Dependencies
pip install -r requirements.txt

streamlit run app.py
http://localhost:8501

