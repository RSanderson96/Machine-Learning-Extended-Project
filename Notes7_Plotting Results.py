# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:13:47 2020

@author: b9054751
"""

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


path ='CNN results.csv'
CNN_Results = pd.read_csv(path)

path ='Logistic results.csv'
Logistic = pd.read_csv(path)

path ='Forest results.csv'
Random_Forest = pd.read_csv(path)

path ='MLP results.csv'
MLP_Results= pd.read_csv(path)


# library
import matplotlib.pyplot as plt

f,ax=plt.subplots(2,2,figsize=(15,15))
Logistic.plot(x="Activity", y=["Accuracy", "Precision", "Recall", "F1"], kind="bar", ax = ax[0,0])
ax[0,0].set_title('Logistics',fontsize=12)
ax [0,0].set_ylabel ("Score")
ax [0,0].set_xlabel ("Data Sample")
ax[0,0].set(ylim=(0, 1))
plt.xticks(rotation=30)
Random_Forest.plot(x="Activity", y=["Accuracy", "Precision", "Recall", "F1"], kind="bar", ax = ax[0,1])
ax[0,1].set_title('Random Forest Results',fontsize=12)
ax [0,1].set_ylabel ("Score")
ax [0,1].set_xlabel ("Data Sample")
ax[0,1].set(ylim=(0, 1))
plt.xticks(rotation=30)
MLP_Results.plot(x="Activity", y=["Accuracy", "Precision", "Recall", "F1"], kind="bar", ax = ax[1,0])
ax[1,0].set_title('MLP Results',fontsize=12)
ax [1,0].set_ylabel ("Score")
ax [1,0].set_xlabel ("Data Sample")
ax[1,0].set(ylim=(0, 1))
plt.xticks(rotation=30)
CNN_Results.plot(x="Activity", y=["Accuracy", "Precision", "Recall", "F1"], kind="bar", ax = ax[1,1])
ax[1,1].set_title('CNN Results',fontsize=12)
ax [1,1].set_ylabel ("Score")
ax [1,1].set_xlabel ("Data Sample")
ax[1,1].set(ylim=(0, 1))
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

