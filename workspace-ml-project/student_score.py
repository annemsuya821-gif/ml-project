#Importing Libraries
#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#STEP 1: DATA PREPROCESSING
dataset = pd.read_csv('workspace-ml-project/studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0)

#STEP 2: FITTING SIMPLE LINEAR REGRESSION INTO THE TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

#STEP 3:PREDICTING
Y_pred = regressor.predict(X_test)

#STEP 4:VISUALIZATION
#Training  Results
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Training Set Result")

plt.show()
plt.savefig("graph.png")

#Testing Results
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Test Set Result")

plt.show()
plt.savefig("graph.png")
