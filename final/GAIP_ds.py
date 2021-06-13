"""Objective of the problem: The objective of the problem is to predict values “buying price” attribute from the given 
features of the Test data. The predictions are to be written to a CSV file along with ID which is the unique 
identifier for each tuple. Please view the sample submission file to understand how the submission file is to 
be written. Please upload the submission file to get a score.

Description of files:
• V.id: Unique vehicle id
• On road old: On road price of the vehicle when purchased from showroom in rupees
• New on road: new on road price in rupees
• Years: Vehicles age
• km: total distance covered by vehicle in km
• Rating: Overall rating of the new vehicle out of 5
• Condition: current condition of the vehicle out of 10(note :- higher the number better the condition is)
• Economy: current fuel economy of the vehicle per liter.
• Top speed: current top speed of the vehicle indicated by dyno test.
• Hp: horse power of the engine indicated by dyno test.
• Torque: torque of the engine indicated by dyno test.
• buying price: predicted price of the vehicle .
Profit: profit made after selling the vehicle
Evaluation Criteria: Normalised root of MSE . 
All values would be normalised to 100. 5 submissions are allowed per person. 
All must submit their projects to the code submission stage."""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#%%
train = pd.read_csv("C:\\Users\\sahas\\Downloads\\cars.csv")
test = pd.read_csv("C:\\Users\\sahas\\Downloads\\test.csv")
#%%

train.info()
#%%
train.describe()
#%%
train.head(20)
cols = train.shape[1]
#%%
reg= linear_model.LinearRegression()
train.hist(bins=50, figsize=(20,15))
plt.show()

#%%
from pandas.tools.plotting import scatter_matrix
attributes = ["on road old","on road now", "years","buying price"]
scatter_matrix(train[attributes], figsize=(20,12))

#%%
#fitting the linear model with 'on road old' and 'buying price'
X_train = train.iloc[:,[2]]
Y_train = train.iloc[:,[11]]
#%%
reg.fit(X_train, Y_train)
#%%
X_test = test.iloc[:,[1]]
Y_test = pd.DataFrame(reg.predict(X_test))
#testing without k-fold cross validation
#%%

from sklearn.model_selection import cross_val_score
scores = cross_val_score(reg, X_train, Y_train, scoring="neg_mean_squared_error", cv=5)
rmse = np.sqrt(-scores)

rmse.mean()
#%%
rmse.std()

#%%
Y_test.to_csv(r'C:\\Users\\sahas\\Downloads\\LinModel2.csv', index=True)


