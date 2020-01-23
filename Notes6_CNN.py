import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplt
from sklearn import tree
from sklearn import metrics 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K

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


##CNN

#testing and training samples

Train=train_ts
Test = test_ts

x_test = np.delete((Test), 18, 1) 
y_test = Test [:,18]
x_train = np.delete((Train), 18, 1)
y_train = Train[:,18]
num_classes = 2

#correct array shape
x_train = x_train.reshape(x_train.shape[0],18,1)
x_test = x_test.reshape(x_test.shape[0], 18, 1)

#final shape of the x samples
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
from keras.utils import to_categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#building the layers of the model - parameters can be varied
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=50, kernel_size=3, activation='relu', input_shape=(18,1)))
model_cnn.add(Conv1D(filters=50, kernel_size=3, activation='relu')) 
model_cnn.add(MaxPooling1D(pool_size=3))
model_cnn.add(Dropout(0.5))
model_cnn.add(Flatten())
model_cnn.add(Dense(2, activation='sigmoid')) #sigmoid - findings do not need to add to 1
model_cnn.summary()


callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
]

model_cnn.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 40 #number of training examples
EPOCHS = 20 #number of repeats of the algoritm

history = model_cnn.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

test_loss, test_acc = model_cnn.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)


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


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# predict probabilities for test set
cnn_probs = model_cnn.predict(x_test, verbose=1)
cnn_classes = model_cnn.predict_classes(x_test, verbose=1)
#cnn_classes = cnn_classes.astype('float32')

y_test = Test [:,18]

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, cnn_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, cnn_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, cnn_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, cnn_classes)
print('F1 score: %f' % f1)



###########################################
Train=train_ts[train_ts[:,13]==1]
Test = test_ts [test_ts[:,13]==1]


x_test = np.delete((Test), 18, 1) 
y_test = Test [:,18]
x_train = np.delete((Train), 18, 1)
y_train = Train[:,18]
num_classes = 2

#correct array shape
x_train = x_train.reshape(x_train.shape[0],18,1)
x_test = x_test.reshape(x_test.shape[0], 18, 1)

#final shape of the x samples
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
from keras.utils import to_categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#building the layers of the model - parameters can be varied
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=50, kernel_size=3, activation='relu', input_shape=(18,1)))
model_cnn.add(Conv1D(filters=50, kernel_size=3, activation='relu')) 
model_cnn.add(MaxPooling1D(pool_size=3))
model_cnn.add(Dropout(0.5))
model_cnn.add(Flatten())
model_cnn.add(Dense(2, activation='sigmoid')) #sigmoid - findings do not need to add to 1
model_cnn.summary()


callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
]

model_cnn.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 40 #number of training examples
EPOCHS = 200 #number of repeats of the algoritm

history = model_cnn.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

test_loss, test_acc = model_cnn.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)


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


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# predict probabilities for test set
cnn_probs = model_cnn.predict(x_test, verbose=1)
cnn_classes = model_cnn.predict_classes(x_test, verbose=1)
#cnn_classes = cnn_classes.astype('float32')

y_test = Test [:,18]

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, cnn_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, cnn_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, cnn_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, cnn_classes)
print('F1 score: %f' % f1)


!git commit -m "Adding accuracy scores"