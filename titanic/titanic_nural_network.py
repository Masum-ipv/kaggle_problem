import numpy as np 
import tflearn as tf
import pandas as pd


#Load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
print(train_data.head(10))

# Delete Name column from datasets (No need for them in the analysis)
del train_data['Name']
del test_data['Name']

# Delete Ticket column from datasets  (No need for them in the analysis)
del train_data['Ticket']
del test_data['Ticket']

print(train_data.head(10))