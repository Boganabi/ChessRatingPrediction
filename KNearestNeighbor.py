# data visualization modules
import pandas as pd
import numpy as np

# scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# import the csv file
dataset = pd.read_csv("games.csv")

# init the preprocessing module
le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()

# split the data into training and testing data
x = dataset.drop(columns=['white_rating', 'black_rating'], axis=1)
y = dataset['white_rating'].values
y2 = dataset['black_rating'].values

# preprocess data to make sure KNN can process the data
enc.fit(x)
x = enc.transform(x).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# define k and fit the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

p = knn.predict(x_test)

print("Score for predicting white's rating:")
score = accuracy_score(y_test, p)

print(score)

# calculate for predicting black's rating
x_train, x_test, y2_train, y2_test = train_test_split(x, y2, test_size=0.5)
knn.fit(x_train, y2_train)

p2 = knn.predict(x_test)

print("Score for predicting black's rating:")
score2 = accuracy_score(y2_test, p2)

print(score2)