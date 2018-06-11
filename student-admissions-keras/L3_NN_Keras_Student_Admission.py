#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 18:21:48 2018

@author: JohnFreeman
"""

#### Predicting Student Admissions with Neural Network in Keras
### Predict student admissions to graduate school at UCLA based on three pieces of data
### 1. GRE Test (Test)
### 2. GPA Scores (Scores)
### 3. Class Rank (1-4)

## (1) Loading the Data
# Importing pandas and numpy
import pandas as pd
import numpy as np

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('student_data.csv')

# Printing out the first 10 rows of our data
data[:10]


## (2) Plotting the Data
# Importing matplotlib
import matplotlib.pyplot as plt

# Function (==performs task) to help us plot
def plot_points(data):
    X = np.array(data[["gre", "gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
    
# Plotting the points
plot_points(data)
plt.show()

#^^그래프를 보니 데이터가 제대로 나눠지지 않는다
#^^더 잘나누어 지게 4개의 그래프로 나눠보자
# Separating the ranks
data_rank1 = data[data["rank"]==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]

# Plotting the graphs
plot_points(data_rank1)
plt.title("Rank 1")
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()

#^^ Now it seems that the lower the rank, the higher the acceptance rate.
# Let's use the Rank as one of our inputs.


##(3) One-Hot Encoding the Rank
#^^get_dummies function 을 사용해보자 (pd)
# Make dummy variables for rank
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)

# Drop the previous rank column
one_hot_data = one_hot_data.drop('rank', axis=1)

# Print the first 10 rows of our data
one_hot_data[:10]


##(4) Scaling the Data into a range 0-1
# Copying our data
processed_data = one_hot_data[:]

# Scaling the columns
processed_data['gre'] = processed_data['gre']/800
processed_data['gpa'] = processed_data['gpa']/4.0
processed_data[:10]



##(4) Splitting the data into Training and Testing
#^^ 우리의 알고리즘을 테스트 하기 위해 나눔.
#^^ 10% of total data = Testing Set
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:10])
print(test_data[:10])



##(5) Splitting the Data into Features and Targets (Labels)
#^^ Final step before the training
#^^ Also, need to one-hot encode the output ^^ with "to_categorical" function
#^^ install keras and tensorflow in Anaconda from Terminal
#^^ >>> pip install keras tensorflow
import keras

# Separate data and one-hot encode the output
# Note: We're also turning the data into numpy arrays, in order to train the model in Keras
features = np.array(train_data.drop('admit', axis=1))
targets = np.array(keras.utils.to_categorical(train_data['admit'], 2))
features_test = np.array(test_data.drop('admit', axis=1))
targets_test = np.array(keras.utils.to_categorical(test_data['admit'], 2))

print(features[:10])
print(targets[:10])



##(6) Defining the Model Architecture
# ^^This is how we use Keras to build our Neural Network
# Imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Builidng the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(6,)))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(2, activation='softmax'))

# Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()



##(7) Training the Model
# Training the model
model.fit(features, targets, epochs=200, batch_size=100, verbose=0)



##(8) Scoring the Model
# Evaluating the model on the training and testing set
score = model.evaluate(features, targets)
print("\n Training Accuracy:", score[1])
score = model.evaluate(features_test, targets_test)
print("\n Testing Accuracy:", score[1])


##(9) Challenge: Play with the Parameters!


