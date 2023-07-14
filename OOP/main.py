import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

class Model:
    def __init__(self, datafile = "../LinearRegression/Data/insurance.csv"):
        self.df = pd.read_csv(datafile)
        self.linear_reg = LinearRegression()

if __name__ == '__main__':
    model_instance = Model()
    print(model_instance.df.head())
    print(model_instance.linear_reg)

    