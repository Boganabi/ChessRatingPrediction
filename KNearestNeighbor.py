# data visualization modules
import pandas as pd
import numpy as np

# scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, accuracy_score

# import the csv file
dataset = pd.read_csv("games.csv")

# get y values before dataset drops them
y = dataset['white_rating'].values
y2 = dataset['black_rating'].values

def manualPreprocess():
    # drop columns id, white_id, black_id, opening_name, moves
    temp = dataset.drop(columns=['white_rating', 'black_rating', 'id', 'created_at', 'last_move_at', 'white_id', 'black_id', 'opening_name', 'moves'])

    index = 0 # for keeping track of index we are editing df at
    # column rated will be 0 = false and 1 = true
    for item in temp['rated'].values:
        if item == "False":
            temp.loc[index, 'rated'] = "0"
        if item == "True":
            temp.loc[index, 'rated'] = "1"
        index = index + 1

    index = 0
    # column victory_status will be coded like outoftime = 1, resign = 2, mate = 3, draw = 4
    for item in temp['victory_status'].values:
        if item == "outoftime":
            temp.loc[index, 'victory_status'] = "1"
        if item == "resign":
            temp.loc[index, 'victory_status'] = "2"
        if item == "mate":
            temp.loc[index, 'victory_status'] = "3"
        if item == "draw":
            temp.loc[index, 'victory_status'] = "4"
        index = index + 1

    index = 0
    # column winner is coded like white = 0, black = 1
    for item in temp['winner'].values:
        if item == "white":
            temp.loc[index, 'winner'] = "0"
        if item == "black":
            temp.loc[index, 'winner'] = "2"
        if item == "draw":
            temp.loc[index, 'winner'] = "1"
        index = index + 1

    index = 0
    # column increment_code will just remove the + between the 2 numbers
    for item in temp["increment_code"].values:
        temp.loc[index, 'increment_code'] = (str(item).replace("+", ""))
        index = index + 1

    index = 0
    # column opening eco will change the first letter to the corresponding number in the alphabetical order
    # the first letter is a letter A-E, so we only need to account for those letters
    for item in temp['opening_eco'].values:
        if item[0] == "A":
            temp.loc[index, 'opening_eco'] = "0" + (item[1]) + (item[2])
        if item[0] == "B":
            temp.loc[index, 'opening_eco'] = "1" + (item[1]) + (item[2])
        if item[0] == "C":
            temp.loc[index, 'opening_eco'] = "2" + (item[1]) + (item[2])
        if item[0] == "D":
            temp.loc[index, 'opening_eco'] = "3" + (item[1]) + (item[2])
        if item[0] == "E":
            temp.loc[index, 'opening_eco'] = "4" + (item[1]) + (item[2])
        index = index + 1

    return temp

def automaticPreprocess():
    # init the preprocessing module
    # le = preprocessing.LabelEncoder()
    enc = preprocessing.OneHotEncoder()
    # x = dataset.drop(columns=['white_rating', 'black_rating'], axis=1)
    x = dataset
    enc.fit(x)
    x = enc.transform(x).toarray()
    return x

# preprocess data to make sure KNN can process the data

# x = manualPreprocess()
x = automaticPreprocess()

# the below will call both auto and manual
# dataset = manualPreprocess()
# x = automaticPreprocess()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# define k and fit the model
knn = KNeighborsClassifier(n_neighbors=3000)
knn.fit(x_train, y_train)

p = knn.predict(x_test)

print("Score for predicting white's rating (0 is best):")
score = mean_squared_error(y_test, p)

print(np.sqrt(score))

# calculate for predicting black's rating
x_train, x_test, y2_train, y2_test = train_test_split(x, y2, test_size=0.5)
knn.fit(x_train, y2_train)

p2 = knn.predict(x_test)

print("Score for predicting black's rating (0 is best):")
score2 = mean_squared_error(y2_test, p2)

print(np.sqrt(score2))