# data visualization modules
import pandas as pd
import numpy as np

# scikit-learn modules
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import the csv file
dataset = pd.read_csv("games.csv")

# get x and y values
x = dataset.drop(columns=['white_rating', 'black_rating', 'id', 'created_at', 'last_move_at', 'white_id', 'black_id', 'opening_name', 'moves'])
y = dataset['white_rating'].values
y2 = dataset['black_rating'].values

# preprocess data
enc = preprocessing.OneHotEncoder()
# x = dataset.drop(columns=['white_rating', 'black_rating'], axis=1)
# x = dataset
enc.fit(x)
# x = enc.transform(x).toarray()
x = enc.transform(x)

poly = PolynomialFeatures(degree=2, include_bias=False)

pf = poly.fit_transform(x)

# using parameter random_state=42 helps us get consistant results
xtr, xte, ytr, yte = train_test_split(pf, y, test_size=0.5, random_state=42)

# polynomial regression is built on top of linear regression, we defined the degree above so its not actually linear :)
prm = LinearRegression()

print("Error for predicting white's rating:")

prm.fit(xtr, ytr)
predicted = prm.predict(xte)

error = np.sqrt(mean_squared_error(yte, predicted))
print(error)

print("Error for predicting Black's rating:")

xtr, xte, ytr, yte = train_test_split(pf, y2, test_size=0.3, random_state=42)

prm.fit(xtr, ytr)
predicted = prm.predict(xte)

error = np.sqrt(mean_squared_error(yte, predicted))
print(error)