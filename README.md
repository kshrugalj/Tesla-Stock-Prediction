Tesla Stock Prediction

This project applies supervised machine learning models to predict short-term stock price movements of Tesla (TSLA) using historical data.

⸻

Overview

The objective is to determine whether Tesla’s stock closing price will increase or decrease the next day. The project uses engineered features from the dataset and compares multiple machine learning models to evaluate prediction accuracy.

⸻

Dataset

The dataset is a CSV file (TSLA.csv) containing historical Tesla stock prices with the following columns:
	•	Date
	•	Open, High, Low, Close, Adj Close
	•	Volume

Preprocessing steps:
	•	Removed the redundant Adj Close column.
	•	Converted the Date column to datetime format.
	•	Extracted year, month, day, and quarter-end indicators.
	•	Engineered new features:
	•	open-close: difference between opening and closing prices
	•	low-high: difference between low and high prices
	•	is_quarter_end: indicator for whether the date is at the end of a financial quarter

The target variable is defined as:

target = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

	•	1: next day’s closing price is higher
	•	0: next day’s closing price is lower or unchanged

⸻

Features Used

The model is trained using three features:
	1.	open-close
	2.	low-high
	3.	is_quarter_end

All features are standardized with StandardScaler before training.

⸻

Models Implemented

The following models are trained and evaluated:
	•	Logistic Regression
	•	Support Vector Classifier (Polynomial Kernel)
	•	XGBoost Classifier

Evaluation metrics include:
	•	Training ROC-AUC
	•	Validation ROC-AUC
	•	Confusion matrix

⸻

Results

The models generally achieve high recall for predicting “up” days, but show a tendency to misclassify “down” days, indicating bias toward predicting an increase.

Example confusion matrix for Logistic Regression:

	Predicted 0	Predicted 1
True 0	1	139
True 1	2	154

Interpretation:
	•	Strong recall for “up” days (True 1).
	•	Low precision due to many false positives.

⸻

Visualizations (Optional)

The code includes several exploratory and analytical visualizations:
	•	Boxplots of stock price distributions
	•	Yearly average trends for Open, High, Low, and Close
	•	Pie chart of target distribution
	•	Heatmap of feature correlations
	•	Tesla Close Price time series plot

⸻

Technologies Used
	•	Python
	•	Pandas
	•	NumPy
	•	Matplotlib
	•	Seaborn
	•	Scikit-learn
	•	XGBoost

⸻

How to Run
	1.	Clone the repository or download the project files.
	2.	Ensure TSLA.csv is located in the project directory.
	3.	Install dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn xgboost


	4.	Run the Python script or Jupyter Notebook.

⸻

Project Structure

Tesla-Stock-Prediction/
│
├── TSLA.csv                # Historical Tesla stock data
├── main.py                 # Main Python script with preprocessing, training, and evaluation
├── README.md               # Project documentation
└── notebooks/              # (Optional) Jupyter notebooks for exploration and visualization


⸻

Notes
	•	This is a binary classification problem, not a price forecasting task.
	•	Predictions are based on a small set of engineered features and should not be used for actual trading decisions.
	•	Adding additional data and indicators could improve model performance.

⸻

Future Improvements
	•	Add technical indicators such as moving averages, RSI, MACD, or Bollinger Bands.
	•	Incorporate sentiment analysis from financial news or social media.
	•	Explore deep learning models such as LSTMs, GRUs, or Transformers for sequential data.
	•	Perform hyperparameter tuning with GridSearchCV or Bayesian optimization.
	•	Address class imbalance using SMOTE, class weights, or threshold adjustments.
	•	Implement backtesting to simulate and evaluate trading strategies.
