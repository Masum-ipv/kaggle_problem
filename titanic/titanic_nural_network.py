import numpy as np
import tflearn
import pandas as pd

#Load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
print('Train dataset: %s, test data %s' %(str(train_data.shape), str(test_data.shape)))


#Check for missing data & list them 
# nas = pd.concat([train_data.isnull().sum(), test_data.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset'], sort=True)
# print('Nan in the data sets')
# print(nas[nas.sum(axis=1) > 0])


# cLEANING THE DATASETS
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
test_data['Age'] = test_data['Age'].astype(int)

# Fare
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)


# Feature that tells whether a passenger had a cabin on the Titanic
train_data['Has_Cabin'] = train_data["Cabin"].apply(lambda x:0 if type(x) == float else 1)
test_data['Has_Cabin'] = test_data["Cabin"].apply(lambda x:0 if type(x) == float else 1)

# Group them
full_dataset = [train_data, test_data]

#Add family size feature
for dataset in full_dataset:
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize
for dataset in full_dataset:
	dataset['IsAlone'] = 0
	dataset.loc[dataset['FamilySize'] ==1, 'IsAlone'] = 1

for dataset in full_dataset:
	dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)

for dataset in full_dataset:    
    dataset.loc[ dataset['Age'] <= 14, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

for dataset in full_dataset:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


# Delete Name column from datasets (No need for them in the analysis)
del train_data['Name']
del test_data['Name']

del train_data['SibSp']
del test_data['SibSp']

del train_data['Parch']
del test_data['Parch']

del train_data['FamilySize']
del test_data['FamilySize']

del train_data['Cabin']
del test_data['Cabin']

# Delete Ticket column from datasets  (No need for them in the analysis)
del train_data['Ticket']
del test_data['Ticket']

del train_data['Embarked']
del test_data['Embarked']

print "=============== After Cleaning Data ========================="
print train_data.head(10)

print('train dataset: %s, test dataset %s' %(str(train_data.shape), str(test_data.shape)) )

del train_data['PassengerId']
X_train = train_data.drop("Survived",axis=1).as_matrix()
Y_train = train_data["Survived"].as_matrix().reshape((891,1))
X_test  = test_data.drop("PassengerId",axis=1).as_matrix()

print "X_train.shape", X_train
print "Y_train.shape", Y_train.shape
print "X_test.shape", X_test.shape


# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 1, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(X_train, Y_train, n_epoch=10, batch_size=16, show_metric=True)

# # Let's create some data for DiCaprio and Winslet
# dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
# winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
# # Preprocess data
# dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# # Predict surviving chances (class 1 results)
pred = model.predict(X_test)
print pred
# print("DiCaprio Surviving Rate:", pred[0][1])
# print("Winslet Surviving Rate:", pred[1][1])