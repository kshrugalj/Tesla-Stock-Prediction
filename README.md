# Tesla Stock Prediction

This project applies supervised machine learning models to predict the short-term price movement of Tesla (TSLA) stock using historical data.

---

## 📈 Overview

The main objective is to build a binary classification model that predicts whether Tesla’s stock closing price will **increase** or **decrease** on the next trading day. The project focuses on feature engineering and a comparative analysis of several machine learning models to evaluate their predictive accuracy.

---

## 💾 Dataset

The model is trained on a CSV file (`TSLA.csv`) containing historical daily stock data for Tesla.

**Columns:**
* `Date`
* `Open`, `High`, `Low`, `Close`, `Adj Close`
* `Volume`

### Preprocessing

1.  **Data Cleaning:** The redundant `Adj Close` column was removed.
2.  **Date Handling:** The `Date` column was converted to a `datetime` object to extract time-based features like year, month, and day.
3.  **Feature Engineering:** New features were created to capture daily price volatility and market behavior:
    * `open-close`: The difference between the opening and closing prices.
    * `low-high`: The difference between the low and high prices for the day.
    * `is_quarter_end`: A binary flag indicating if the date is the last day of a financial quarter.

### Target Variable

The target variable is a binary indicator of the next day's price movement.
* **1**: The next day’s closing price is **higher** than the current day's.
* **0**: The next day’s closing price is **lower or unchanged**.

```python
target = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
