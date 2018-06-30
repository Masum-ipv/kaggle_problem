# https://www.kaggle.com/anaskad/step-by-step-solving-titanic-problem/notebook

import pandas as pd
import numpy as np
import csv as csv


#Load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
print('Train dataset: %s, test data %s' %(str(train_data.shape), str(test_data.shape)))
print(train_data.head(10))


#Check for missing data & list them 
nas = pd.concat([train_data.isnull().sum(), test_data.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset'], sort=True)
print('Nan in the data sets')
print(nas[nas.sum(axis=1) > 0])


# Data sets cleaing, fill nan (null) where needed and delete uneeded columns
#manage Age
train_random_ages = np.random.randint(train_data["Age"].mean() - train_data["Age"].std(),
                                          train_data["Age"].mean() + train_data["Age"].std(),
                                          size = train_data["Age"].isnull().sum())

test_random_ages = np.random.randint(test_data["Age"].mean() - test_data["Age"].std(),
                                          test_data["Age"].mean() + test_data["Age"].std(),
                                          size = test_data["Age"].isnull().sum())



train_data["Age"][np.isnan(train_data["Age"])] = train_random_ages
test_data["Age"][np.isnan(test_data["Age"])] = test_random_ages
train_data['Age'] = train_data['Age'].astype(int)
test_data['Age']    = test_data['Age'].astype(int)

# Embarked 
train_data["Embarked"].fillna('S')
train_data["Embarked"].fillna('S', inplace=True)
train_data['Port'] = train_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_data['Port'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
del train_data['Embarked']
del test_data['Embarked']

# Fare
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)

print train_data.head(10)