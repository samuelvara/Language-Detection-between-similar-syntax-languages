import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#%%
train = pd.read_csv("C:\\Users\\sahas\\Downloads\\cars.csv")
test = pd.read_csv("C:\\Users\\sahas\\Downloads\\test.csv")

#%%
train.hist(bins=50, figsize=(20,15))
plt.show()

#%%
from pandas.tools.plotting import scatter_matrix
attributes = ["on road old","on road now", "years","buying price"]
scatter_matrix(train[attributes], figsize=(20,12))

#%%
#fitting the linear model with 'on road old' VS 'buying price','years' and 'km'
X_train = train.iloc[:,[1,3,4]]
Y_train = train.iloc[:,[11]]
#%%
X_tester = test.iloc[:,[1,3,4]]
X_tester.head()

#%%
reg= LinearRegression()
reg.fit(X_train, Y_train)
Y_out = pd.DataFrame(reg.predict(X_tester))
Y_out.to_csv('C:\\Users\\sahas\\Downloads\\LinOn3.csv', index=True)
