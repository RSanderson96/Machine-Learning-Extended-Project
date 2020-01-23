# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:46:52 2020

@author: b9054751
"""

import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
from glob import glob
from pandas import Series
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


def get_ds_infos():
    ## 0:Code, 1:Weight, 2:Height, 3:Age, 4:Gender
    dss = np.genfromtxt("data_subjects_info.csv",delimiter=',')
    dss = dss[1:]
    print("----> Data subjects information is imported.")
    return dss #read data subjects CSV file
ds_list = get_ds_infos()

def create_time_series(num_features, num_act_labels, num_gen_labels, label_codes, trial_codes):
    dataset_columns = num_features+num_act_labels+num_gen_labels
    ds_list = get_ds_infos() #information about the participants
    train_data = np.zeros((0,dataset_columns))
    test_data = np.zeros((0,dataset_columns))
    for i, sub_id in enumerate(ds_list[:,0]):
        for j, act in enumerate(label_codes):
            for trial in trial_codes[act]:
                fname = 'motionsense-dataset/A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1) #drop first column
                unlabel_data = raw_data.values #need to lavel the data for training
                label_data = np.zeros((len(unlabel_data), dataset_columns))
                label_data[:,:-(num_act_labels + num_gen_labels)] = unlabel_data
                label_data[:,label_codes[act]] = 1 #if that activity is included, 1 in column
                label_data[:,-(num_gen_labels)] = int(ds_list[i,4]) #picking the gender column
                ## long trials = training dataset Short trials = test dataset
                if trial > 10: #training data is later trials
                    test_data = np.append(test_data, label_data, axis = 0)
                else:    
                    train_data = np.append(train_data, label_data, axis = 0)
    return train_data , test_data #returns the two datasets - rows of individual results for each time stamp
print("--> Start...")
## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
num_features = 12 # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
num_act_labels = 6 # dws, ups, wlk, jog, sit, std
num_gen_labels = 1 # 0/1(female/male)
label_codes = {"dws":num_features, "ups":num_features+1, "wlk":num_features+2, "jog":num_features+3, "sit":num_features+4, "std":num_features+5}
trial_codes = {"dws":[1,2,11], "ups":[3,4,12], "wlk":[7,8,15], "jog":[9,16], "sit":[5,13], "std":[6,14]}    
## Calling 'creat_time_series()' to build time-series
print("--> Building Training and Test Datasets...")
train_ts, test_ts = create_time_series(num_features, num_act_labels, num_gen_labels, label_codes, trial_codes)
print("--> Shape of Training Time-Seires:", train_ts.shape)
print("--> Shape of Test Time-Series:", test_ts.shape)

#final dataframe - column = measurement/activity/characteristic
#                   row = the information about the specific timestamp for each person


#############
#(need background information for the sample)
#distributions - height/weight/gender/age boxplots. 

#dataset[['weight','height','age']].boxplot()
#fig = plt.gcf()
#fig.set_size_inches(30, 15)

#time series graphs?

##________________________________
## For Example: Attiude data
## female
data = train_ts[train_ts[:,-1]==0]
## jogging
data = data[data[:,-4]==1]
## 10 seconds
data = pd.DataFrame(data[10000:10500,0:3])
data.plot()
plt.xlabel('Second', fontsize=18)
plt.ylabel('Value', fontsize=16)
lgnd=plt.legend()
lgnd.get_texts()[0].set_text('roll')
lgnd.get_texts()[1].set_text('pitch')
lgnd.get_texts()[2].set_text('yaw')
fig = pyplt.gcf()
fig.set_size_inches(18, 8)
plt.show()

## For Example: Attiude data
## male
data = train_ts[train_ts[:,-1]==1]
## jogging
data = data[data[:,-4]==1]
## 10 seconds
data = pd.DataFrame(data[10000:10500,0:3])
data.plot()
plt.xlabel('Second', fontsize=18)
plt.ylabel('Value', fontsize=16)
lgnd=plt.legend()
lgnd.get_texts()[0].set_text('roll')
lgnd.get_texts()[1].set_text('pitch')
lgnd.get_texts()[2].set_text('yaw')
fig = pyplt.gcf()
fig.set_size_inches(18, 8)
plt.show()





path = 'C:\Temp\OneDrive - Newcastle University\Machine Learning\Extended Project\Data\A_DeviceMotion_data\A_DeviceMotion_data\dws_1\sub_1.csv'
File1 = pd.read_csv(path)
Down_1 = File1

path ='data_subjects_info.csv'
Subject_Info = pd.read_csv(path)
correct_info = Subject_Info

#is there any missing information?
#What is missing?
missing_info = Corrected_Info.isnull().sum().sort_values(ascending = False)
missing_percent = (Corrected_Info.isnull().sum()/Corrected_Info.isnull().count()*100).sort_values(ascending = False)
#there are no missing values

#Need to categorise the information about the participants.
#What is the distribution in the sample?

#WEIGHT
x = Subject_Info.weight
sns.distplot(x)

def process_weight(df,cut_points,label_names):
    correct_info["Weight_categories"] = pd.cut(correct_info["weight"],cut_points,labels=label_names)
    return df #replace the ages with the cutpoints
cut_points = [40, 50, 60, 70, 80, 90, 100, 110] # split into age groups
label_names = ["40-50", "50-60", "60-70", "70-80", "80-90", "90-100", "100-110"]
correct = process_weight(correct_info,cut_points,label_names) #apply formula
correct_info ['Weight_categories'] = correct['Weight_categories'] #apply to original table
correct_info = correct_info.drop(columns = "weight") #remove original weight columns
correct_info['Weight_categories'] = correct_info['Weight_categories'].map({'40-50':0,"50-60":1, "60-70":2, "70-80":3, "80-90":4, "90-100":5, "100-110":6 }).astype(int)
#categorise for training

#repeat for all other columns 

#HEIGHT
x = Subject_Info.height
sns.distplot(x)

def process_height(df,cut_points,label_names):
    correct_info["Height_categories"] = pd.cut(correct_info["height"],cut_points,labels=label_names)
    return df #replace the ages with the cutpoints
cut_points = [160, 165, 170, 175, 180, 185, 190, 195] # split into age groups
label_names = ["160-165", "165-170", "170-175", "175-180", "180-185", "185-190", "190-195"]
correct = process_height(correct_info,cut_points,label_names)
correct_info ['Height_categories'] = correct['Height_categories']
correct_info = correct_info.drop(columns = "height")
correct_info['Height_categories'] = correct_info['Height_categories'].map({"160-165":0, "165-170":1, "170-175":2, "175-180":3, "180-185":4, "185-190":5, "190-195":6 }).astype(int)


#AGE
x = Subject_Info.age
sns.distplot(x)

def process_age(df,cut_points,label_names):
    correct_info["Age_categories"] = pd.cut(correct_info["age"],cut_points,labels=label_names)
    return df #replace the ages with the cutpoints
cut_points = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] # split into age groups
label_names = ["0 - 5", "5- 10", "10 - 15", "15 - 20", "20-25", "25 -30", "30-35",
               "35 - 40", "40-45", "45-50", "50-55", "55-60"]
correct = process_age(correct_info,cut_points,label_names)
correct_info ['Age_categories'] = correct['Age_categories']
correct_info = correct_info.drop(columns = "age")
correct_info['Age_categories'] = correct_info['Age_categories'].map({"0 - 5":0, "5- 10":1, "10 - 15":2, "15 - 20":3, "20-25":4, "25 -30":5, "30-35":6,
               "35 - 40":7, "40-45":8, "45-50":9, "50-55":10, "55-60":11 }).astype(int)



#Gender - already catgorised


#check correlations between groups in the dataset - how good is the sample in general?



#Machine Learning??
test_ts = pd.DataFrame(test_ts)
train_ts = pd.DataFrame(train_ts)

#1) Logistic Regression: Which measurement has the highest F1 Score?
                        #Whch Activity has the highest F1 score?
                        
                    
Train = train_ts[train_ts[17]==1] #set out datasets (in this case, only standing)
Test = test_ts[test_ts[17]==1]                    

lr = LogisticRegression() 
features= [0,1,2,3,4,5,6,7,8,9,10,11]
test_X = Test[features]
lr.fit(Train[features], Train[18])

Log_predictions = lr.predict(test_X)

Y_Predictions = Log_predictions
Y_True = Test[18]

print("Accuracy:",metrics.accuracy_score(Y_True, Y_Predictions)) 

print('Precision: %.3f' % precision_score(Y_True, Y_Predictions))  
print('Recall: %.3f' % recall_score(Y_True, Y_Predictions)) 
print('F1: %.3f' % f1_score(Y_True, Y_Predictions))

#do this for all activities/measurements

################################################
#MLP
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate data
TestDF = pd.DataFrame(test_ts)
TrainDF = pd.DataFrame(train_ts)
features= [0,1,2,3,4,5,6,7,8,9,10,11]
x_test = TestDF[features]
y_test = TestDF[18]
x_train = TrainDF[features]
y_train = TrainDF[18]


model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
predictions = model.predict(x_test)
score = model.evaluate(x_test, y_test, batch_size=128)

#################################

from sklearn.neural_network import MLPClassifier
features= [0,1,2,3,4,5,6,7,8,9,10,11]
x_train = train_ts [:,0:17]
y_train = train_ts[:,18]
x_test = test_ts [:,0:17]
y_test = test_ts[:,18]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(x_train, y_train)
prediction = clf.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, prediction)) #0.63553
print('Precision: %.3f' % precision_score(y_test, prediction)) #0.647 
print('Recall: %.3f' % recall_score(y_test, prediction)) #0.819
print('F1: %.3f' % f1_score(y_test, prediction)) #0.723

#############################################
#Random Forest 
TestDF = pd.DataFrame(test_ts)
TrainDF = pd.DataFrame(train_ts)

features= [0,1,2,3,4,5,6,7,8,9,10,11]
x_test = TestDF[features]
y_test = TestDF [18]
x_train = TrainDF[features]
y_train = TrainDF[18]

Random_Forest = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=1)
Random_Forest.fit(x_train, y_train)
predictions = Random_Forest.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, predictions))#0.66395
print(Random_Forest.score(x_train, y_train)) #0.857402

#Random Forest based on identity data?

x_train, x_test, y_train, y_test = train_test_split(correct_info, correct_info["gender"], test_size=0.3)

Random_Forest = RandomForestClassifier(n_estimators=3, max_depth=10, random_state=1)
Random_Forest.fit(x_train, y_train)
predictions = Random_Forest.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, predictions)) 
print(Random_Forest.score(x_train, y_train))
print('F1: %.3f' % f1_score(
        y_test, predictions))

#################################################################
#
##CNN
#
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


batch_size = 360
num_classes = 2
epochs = 2

x_test = test_ts
y_test = test_ts [:,18]
x_train = train_ts
y_train = train_ts[:,18]

x_train = x_train.reshape(1081446,19,1)
x_test = x_test.reshape(x_test.shape[0], 19, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
from keras.utils import to_categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


from keras.layers import Conv1D, MaxPooling1D,GlobalAveragePooling1D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model_cnn = Sequential()
model_cnn.add(Conv1D(filters=50, kernel_size=3, activation='relu', input_shape=(19,1)))
model_cnn.add(Conv1D(filters=50, kernel_size=3, activation='relu')) 
model_cnn.add(MaxPooling1D(pool_size=3))
model_cnn.add(Dropout(0.5))
model_cnn.add(Flatten())
#model_cnn.add(Dense(100, activation='relu'))
model_cnn.add(Dense(2, activation='sigmoid'))

#model_cnn.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

model_cnn.summary()


callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
]

model_cnn.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 20
EPOCHS = 3

history = model_cnn.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

prediction = model_cnn.predict(x_test)

test_loss, test_acc = model_cnn.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)


prediction = keras.utils.to_categorical(prediction, 2)

predicted_classes = np.argmax(prediction, axis=1)
predicted_classes = predicted_classes.astype('float32')



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




!git commit -m "CNN"




