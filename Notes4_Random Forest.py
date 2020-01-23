# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 08:35:53 2020

@author: b9054751
"""

import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
from pandas import Series
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import confusion_matrix

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





#############################################
#Random Forest 
TestDF = pd.DataFrame(test_ts)
TrainDF = pd.DataFrame(train_ts)

features= range(0,18)
x_test = TestDF[features]
y_test = TestDF [18]
x_train = TrainDF[features]
y_train = TrainDF[18]

Random_Forest = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=1)
Random_Forest.fit(x_train, y_train)
Y_Predictions = Random_Forest.predict(x_test)
Y_True = y_test

print("Accuracy:",metrics.accuracy_score(Y_True, Y_Predictions)) 
print('Precision: %.3f' % precision_score(Y_True, Y_Predictions))  
print('Recall: %.3f' % recall_score(Y_True, Y_Predictions)) 
print('F1: %.3f' % f1_score(Y_True, Y_Predictions))



y_train = TrainDF[18]
features= range(0,18)#which categories should be used?
x_train = TrainDF[features]#training data
x_test = TestDF[features] #testing data
y_train = TrainDF[18]

result_array = np.array([])
for i in range (1,100):
    model = RandomForestClassifier(n_estimators=i, max_depth=5, random_state=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    Y_Predictions = predictions
    result = metrics.accuracy_score(predictions, y_test) 
    result_array = np.append(result_array, result)
Two = result_array

x = Two
y = range(1,100)
plt.scatter(Two,y)




importances = Random_Forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in Random_Forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

Importances = list(zip(features, importances))
print(Importances)
#fixing the graph
Importances = pd.DataFrame(Importances)
Importances = Importances.T

Importances = pd.DataFrame({'features': ["Attitude:Roll", "Attitude: Pitch", "Attitude:Yaw",
                                         "Gravity:X", "Gravity:Y", "Gravity:Z", 
                                         "Rotation_Rate: X","Rotation_Rate: Y", "Rotation_Rate: Z",
                                         "Acceleration: X", "Acceleration: Y","Acceleration: Z", 
                                         "Down", "Up", "Walk", "Jog", "Sit", "Stand"],
    'Importance':[0.08753189159906041, 0.1453880685453445, 0.14514245767614475,
                  0.16079279571400212, 0.11561369180284035, 0.16411946698015445, 
                  0.013978232915168384, 0.008850237364342498, 0.008197712258988892,
                  0.010173443003730666, 0.021433073306058356, 0.043149681236406376,
                  0.007440215251639841, 0.006481200924769619, 0.008545346127933482,
                  0.006345881680639919, 0.027910980838336354, 0.01890562277443911]})
Importances = Importances.sort_values(by = 'Importance', ascending=False)                            

ax = Importances.plot.bar(x="features", y="Importance", title = "Figure Eight: Feature Importances in the Random Forest Classifier")
ax.set_xlabel("Data Features Applied")
ax.set_ylabel("Influence in the Random Forest Classifier")

confusion_matrix(Y_True, Y_Predictions)
sns.heatmap(confusion_matrix(Y_True, Y_Predictions),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix: 7 Variables', y=1.05, size=15)


############################################################

TestDF = pd.DataFrame(test_ts[test_ts[:,12]==1] )
TrainDF = pd.DataFrame(train_ts[train_ts[:,12]==1] )

features= range(0,18)
x_test = TestDF[features]
y_test = TestDF [18]
x_train = TrainDF[features]
y_train = TrainDF[18]

Random_Forest = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=1)
Random_Forest.fit(x_train, y_train)
Y_Predictions = Random_Forest.predict(x_test)
Y_True = y_test

print("Accuracy:",metrics.accuracy_score(Y_True, Y_Predictions)) 
print('Precision: %.3f' % precision_score(Y_True, Y_Predictions))  
print('Recall: %.3f' % recall_score(Y_True, Y_Predictions)) 
print('F1: %.3f' % f1_score(Y_True, Y_Predictions))



importances = Random_Forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in Random_Forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

Importances = list(zip(features, importances))
print(Importances)
#fixing the graph
Importances = pd.DataFrame(Importances)
Importances = Importances.T

Importances = pd.DataFrame({'features': ["Attitude:Roll", "Attitude: Pitch", "Attitude:Yaw",
                                         "Gravity:X", "Gravity:Y", "Gravity:Z", 
                                         "Rotation_Rate: X","Rotation_Rate: Y", "Rotation_Rate: Z",
                                         "Acceleration: X", "Acceleration: Y","Acceleration: Z", 
                                         "Down", "Up", "Walk", "Jog", "Sit", "Stand"],
    'Importance':[ 0.06038412335031429, 0.06913058692734723, 0.09459839645850554,
                  0.12350235921908621, 0.03781150498851057, 0.06485161767765618,
    0.09521044417764003, 0.16819086072347952, 0.04691086274912178, 0.035374124484738804, 
    0.04142095829687499, 0.16261416094672485, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
Importances = Importances.sort_values(by = 'Importance', ascending=False)                            

ax = Importances.plot.bar(x="features", y="Importance", title = "Figure Eight: Feature Importances in the Random Forest Classifier")
ax.set_xlabel("Data Features Applied")
ax.set_ylabel("Influence in the Random Forest Classifier")


confusion_matrix(Y_True, Y_Predictions)
sns.heatmap(confusion_matrix(Y_True, Y_Predictions),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix: 7 Variables', y=1.05, size=15)

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