# Tesla (TSLA) Stock Price Movement Prediction

This project uses machine learning to predict the direction of Tesla's (TSLA) stock price for the next trading day. The goal is to build a binary classification model that predicts whether the closing price will increase (1) or decrease/stay the same (0) compared to the current day's closing price.

***Disclaimer:** This project is for educational and demonstrational purposes only. It is not financial advice.*

## Project Workflow

The project follows a standard machine learning pipeline:

1.  **Data Loading & Exploration**: The historical stock data from `TSLA.csv` is loaded. An initial analysis is performed to understand the dataset's structure, check for missing values, and review descriptive statistics.
2.  **Data Cleaning & Preprocessing**: Redundant columns, such as `Adj Close`, are removed to prevent multicollinearity. The dataset is checked for any null values to ensure data quality.
3.  **Feature Engineering**: To provide more predictive power to the models, several new features are engineered from the existing data:
      * `open-close`: The difference between the opening and closing price of the day, indicating intraday price movement.
      * `low-high`: The difference between the lowest and highest price of the day, representing daily volatility.
      * `is_quarter_end`: A binary flag indicating if the date falls on the end of a financial quarter, which can sometimes influence trading behavior.
      * `target`: The **prediction target**. This is a binary variable that is `1` if the next day's closing price is higher than the current day's, and `0` otherwise.
4.  **Model Training**: The dataset is split into training (90%) and validation (10%) sets. The engineered features are scaled using `StandardScaler` to normalize their range. Three different classification models are trained on this data:
      * **Logistic Regression**: A baseline linear model for binary classification.
      * **Support Vector Classifier (SVC)**: A non-linear model using a polynomial kernel.
      * **XGBoost Classifier**: An efficient and powerful gradient-boosting algorithm.
5.  **Model Evaluation**: The performance of each model is evaluated on the validation set using the **ROC AUC (Area Under the Receiver Operating Characteristic Curve)** score. This metric is well-suited for binary classification as it measures the model's ability to distinguish between the two classes. A confusion matrix is also generated for the Logistic Regression model to visualize its true positive, false positive, true negative, and false negative predictions.

-----

## Technical Details

### Dataset

  * **Source**: `TSLA.csv` containing historical daily stock market data for Tesla Inc.
  * **Key Columns**: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

### Selected Features for Modeling

The final models were trained on the following engineered features:

  * `open-close`
  * `low-high`
  * `is_quarter_end`

### Dependencies

To run this project, you will need Python 3 and the following libraries:

  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `xgboost`
  * `matplotlib`
  * `seaborn`

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

-----

## How to Run the Code

1.  **Clone the repository** and navigate to the project directory.
2.  **Place the dataset** (`TSLA.csv`) in the root folder of the project.
3.  **Execute the Python script** from your terminal:
    ```bash
    python your_script_name.py
    ```
4.  The script will output the training and validation ROC AUC scores for each of the three models and display a confusion matrix for the Logistic Regression model's predictions on the validation data.

## Results

The script trains three models and prints their respective ROC AUC scores for both training and validation sets. This allows for a direct comparison of their performance and helps identify potential overfitting (where training accuracy is significantly higher than validation accuracy).

**Example Output:**

```
LogisticRegression() :
Training Accuracy :  0.528...
Validation Accuracy :  0.519...

SVC(kernel='poly', probability=True) :
Training Accuracy :  0.526...
Validation Accuracy :  0.521...

XGBClassifier(...) :
Training Accuracy :  0.998...
Validation Accuracy :  0.505...
```

The results indicate that while the XGBoost model achieves near-perfect accuracy on the training data, its performance on unseen validation data is close to random chance (an AUC of 0.5), suggesting significant overfitting. The Logistic Regression and SVC models show more stable, albeit modest, predictive performance.
