import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # conda install scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

data = pd.read_csv("Data/insurance.csv")

def understanding_data(data):
    print(data.head())
    print(data.shape)
    print(data.info())
    print(data.isnull())
    print(data.isnull().sum())
    print(data.dtypes)

def preprocessing_data(data):
    data['sex'] = data['sex'].astype('category')
    data['region'] = data['region'].astype('category')
    data['smoker'] = data['smoker'].astype('category')
    # print(data.dtypes)
    # print(data.describe().T)
    return data

def analyse_smoke_data(data):
    smoke_data = data.groupby("smoker").mean().round(2)
    print(smoke_data)

def data_visualization(data):
    sns.set_style("whitegrid")
    sns.pairplot(
        data[["age", "bmi", "charges", "smoker"]],
        hue = "smoker",
        height = 3,
        palette = "Set1")
    sns.heatmap(data.corr(), annot= True)
    plt.show()

def one_hot_encoding(data):
    data = pd.get_dummies(data)  # Converts categorical data to one hot vectors automatically
    print(data.columns)
    return data

def get_train_test_split(data):
    # Features and prediction variable
    y = data["charges"] # We need to predict the insurance charges based on other features
    X = data.drop("charges", axis = 1)
    
    # 80/20 Train Test Split
    X_train,X_test,y_train,y_test=train_test_split(
        X,
        y, 
        train_size = 0.80, 
        random_state = 1)

    return X_train, X_test, y_train, y_test

def regression_model(X_train, X_test, y_train, y_test):
    # Linear Regression model
    print('Running model...')
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    
    print('Model Trained...')
    print('test score: ', lr.score(X_test, y_test).round(3))
    print('train score: ', lr.score(X_train, y_train).round(3)) # If training was high then overfitting

    # Predictions on test data
    y_pred = lr.predict(X_test)

    # RMSE value
    print("RMSE: ", math.sqrt(mean_squared_error(y_test, y_pred))) # This means our value differs from actual value by this amount(Standard Deviation)

    # Model prediction
    data_new = X_train[:1]
    print('Sample data: ', data_new)
    print('Sample prediction: ', lr.predict(data_new))
    print('Real value: ', y_train[:1])

print('Running...')
# understanding_data(data)
data = preprocessing_data(data)
# analyse_smoke_data(data)
# data_visualization(data)
data = one_hot_encoding(data)
print(data.dtypes)
X_train, X_test, y_train, y_test = get_train_test_split(data) 
print('train test split: ', X_train.shape, X_test.shape)
regression_model(X_train, X_test, y_train, y_test)

