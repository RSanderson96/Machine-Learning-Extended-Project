# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 08:36:00 2020

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
from sklearn.neural_network import MLPClassifier
from sklearn import metrics 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score


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


#MLP

TrainDF = pd.DataFrame(train_ts)
TestDF = pd.DataFrame(test_ts)


features= range(0,18)
x_train = TrainDF [features]
y_train = TrainDF[18]
x_test = TestDF [features]
y_test = TestDF[18]
MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 10,10), random_state=1)

MLP.fit(x_train, y_train)
prediction = MLP.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, prediction)) #0.63553
print('Precision: %.3f' % precision_score(y_test, prediction)) #0.647 
print('Recall: %.3f' % recall_score(y_test, prediction)) #0.819
print('F1: %.3f' % f1_score(y_test, prediction)) #0.723

sns.heatmap(confusion_matrix(y_test, prediction),annot=True, annot_kws={"size": 14},fmt='3.0f',cmap="RdBu_r")
plt.title('MLP Confusion_matrix', y=1.05, size=15)








TrainDF = pd.DataFrame(train_ts[train_ts[:,12]==1])
TestDF = pd.DataFrame(test_ts[test_ts[:,12]==1])


features= range(0,18)
x_train = TrainDF [features]
y_train = TrainDF[18]
x_test = TestDF [features]
y_test = TestDF[18]
MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 10,10), random_state=1)

MLP.fit(x_train, y_train)
prediction = MLP.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, prediction)) #0.63553
print('Precision: %.3f' % precision_score(y_test, prediction)) #0.647 
print('Recall: %.3f' % recall_score(y_test, prediction)) #0.819
print('F1: %.3f' % f1_score(y_test, prediction)) #0.723

sns.heatmap(confusion_matrix(y_test, prediction),annot=True, annot_kws={"size": 14},fmt='3.0f',cmap="RdBu_r")
plt.title('MLP Confusion_matrix', y=1.05, size=15)

#
#
#
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import f1_score
#from sklearn.metrics import cohen_kappa_score
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import confusion_matrix
#from keras.models import Sequential
#from keras.layers import Dense
#
#TrainDF = pd.DataFrame(train_ts[train_ts[:,12]==1])
#TestDF = pd.DataFrame(test_ts[test_ts[:,12]==1])
#features= range(0,18)
#x_train = TrainDF [features]
#y_train = TrainDF[18]
#x_test = TestDF [features]
#y_test = TestDF[18]
#
#
#model = Sequential()
#model.add(Dense(100, input_dim=12, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(100, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])
#_, train_acc = model.evaluate(x_train, y_train, verbose=1)
#_, test_acc = model.evaluate(x_test, y_test, verbose=1)
#
#print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
#
##model.fit(x_train, y_train,
##          epochs=20,
##          batch_size=128)
##predictions = model.predict(x_test)
##score = model.evaluate(x_test, y_test, batch_size=128)
#
## plot loss during training
#pyplot.subplot(211)
#pyplot.title('Loss')
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
## plot accuracy during training
#pyplot.subplot(212)
#pyplot.title('Accuracy')
#pyplot.plot(history.history['accuracy'], label='train')
#pyplot.plot(history.history['val_accuracy'], label='test')
#pyplot.legend()
#pyplot.show()
#
#
#model = Sequential()
#model.add(Dense(100, input_dim=12, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(100, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])
#model = model.fit(trainX, trainy, epochs=20, verbose=0)
#
#
## predict probabilities for test set
#yhat_probs = model.predict(testX, verbose=0)
## predict crisp classes for test set
#yhat_classes = model.predict_classes(testX, verbose=0)
#
## predict probabilities for test set
#yhat_probs = model.predict(testX, verbose=0)
#yhat_classes = model.predict_classes(testX, verbose=0)
#yhat_probs = yhat_probs[:, 0]
#yhat_classes = yhat_classes[:, 0]
#
## accuracy: (tp + tn) / (p + n)
#accuracy = accuracy_score(testy, yhat_classes)
#print('Accuracy: %f' % accuracy)
## precision tp / (tp + fp)
#precision = precision_score(testy, yhat_classes)
#print('Precision: %f' % precision)
## recall: tp / (tp + fn)
#recall = recall_score(testy, yhat_classes)
#print('Recall: %f' % recall)
## f1: 2 tp / (2 tp + fp + fn)
#f1 = f1_score(testy, yhat_classes)
#print('F1 score: %f' % f1)
