import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sbclear

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

#Uses path and prints the first 5 rows of data from TSLA.csv
df = pd.read_csv('/Users/kshrugal/Desktop/Machine Learning Practice/Supervised Learning/Tesla Stock Prediction/TSLA.csv')
print(df.head())

#Prints data and info about the CSV file 
print(f"This is the shape {df.shape}")
print(df.describe())
# print(df.info)

#Checking to see if both Close prices and Adj Close prices are the same 
print(f"This shows how many rows and columns are equal {df[df['Adj Close']==df['Close']].shape}")

#Dropping Adj Close and Close because both of the columns are the same, so there is no need to have both of them 
df = df.drop(['Adj Close'], axis = 1)
print(df.head())#Checking to see if Adj Close column was dropped 

#Checking if there any null values 
print(df.isnull().sum())


# features = ['Open', 'High', 'Low', 'Close']
# plt.subplots(figsize=(20,10))
# for i, col in enumerate(features):
#   plt.subplot(2,3,i+1)
#   sbclear.boxplot(df[col])
# plt.show()

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # convert safely to an object 
df['year'] = df['Date'].dt.year #makes year column
df['month'] = df['Date'].dt.month #makes month column
df['day'] = df['Date'].dt.day #makes day column
df['is_quarter_end'] = np.where(df['month']%3==0,1,0) #makes sure if month is end of the quarter, 1 = quarter end, 0 = not quarter end

# data_grouped = df.drop('Date', axis=1).groupby('year').mean()
# plt.subplots(figsize=(20,10))

# for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
#   plt.subplot(2,2,i+1)
#   data_grouped[col].plot.bar()
# plt.show()

print(df.drop('Date', axis=1).groupby('is_quarter_end').mean())#Dropping Date Column and grouping by that quarter_end column and taking mean

#Creating 2 more columns that tells whether last day's price was higher or not 
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)#Decision if we should sell or not, will be trained in ML model 

print(df.head(50))

#Shows pie-chart
# plt.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
# plt.show()

#Makes heatmap to see where it is heavily correlated, shows that 
# plt.figure(figsize=(10,10))
# sbclear.heatmap(df.drop('Date', axis=1).corr()>0.9, annot=True, cbar=False)
# plt.show()

features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

models = [LogisticRegression(), SVC(
  kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(
    Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()
  
ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid)
plt.show()


# #Using matplotlib to graph the TESLA close price data 
# plt.figure(figsize=(15,5))
# plt.plot(df['Close'])#which values we are graphing 
# plt.title('Tesla Close price')#title
# plt.ylabel('Price in dollars')#y-axis label
# plt.show()

