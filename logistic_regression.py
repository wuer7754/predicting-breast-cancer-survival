# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 22:53:18 2022

@author: xia shufan
"""

import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
data_prefix = Path(r"C:\Users\xia shufan\Documents\SNF-Data01\InputData\Breast")
data1_path =  data_prefix.joinpath("Breast_patient_data.csv")
data2_path = data_prefix.joinpath("Breast_gene_expr.csv")
data3_path = data_prefix.joinpath("Breast_rna_expr.csv")
data4_path = data_prefix.joinpath("Breast_type_data.csv")


data1 = pd.read_csv(data1_path,index_col = 0).dropna()
data2 = pd.read_csv(data2_path,index_col = 0).dropna()
data3 = pd.read_csv(data3_path,index_col = 0).dropna()
data4 = pd.read_csv(data4_path,index_col = 0).dropna()


from sklearn.model_selection import train_test_split
model_data = pd.concat([data1, data2, data3, data4], axis = 1) # (1904,820)


xtrain , xtest , ytrain ,ytest = train_test_split(model_data, data4, test_size = 0.3 , random_state= 420)

xtrain.shape # (1332, 819)
xtest.shape # (572, 819)
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
lrl1 = LR(penalty = "l1", solver = "liblinear" , C = 1, max_iter = 1000)
lrl2 = LR(penalty = "l2", solver = "liblinear" , C = 1, max_iter = 1000)
lrl1 = lrl1.fit(xtrain, ytrain)
lrl1.coef_
ytest.value_counts()
xtest.shape
# 0.0    173
# 1.0    178
# 2.0    142
# 3.0    70
# 4.0    9  
accuracy_score(lrl1.predict(xtrain), ytrain)
accuracy_score(lrl1.predict(xtest), ytest)
print("Classification Report : \n",classification_report(lrl1.predict(xtest),ytest))
#                   precision    recall  f1-score   support

#          0.0       1.00      1.00      1.00       173
#          1.0       0.79      0.73      0.76       192
#          2.0       0.55      0.60      0.58       129
#          3.0       0.79      0.71      0.74        78
#          4.0       0.00      0.00      0.00         0

#     accuracy                           0.78       572
#    macro avg       0.63      0.61      0.62       572
# weighted avg       0.80      0.78      0.79       572
print("Confusion Metrix : \n",confusion_matrix(lrl1.predict(xtest),ytest))
# Confusion Metrix : 
#  [[173   0   0   0   0]
#  [  0 141  49   2   0]
#  [  0  37  78  13   1]
#  [  0   0  15  55   8]
#  [  0   0   0   0   0]]



# l1 = []
# l2 = []
# l1test = []
# l2test = []

# for i in np.linspace(0.05, 1, 19): # 在0~1 之间有19个点
#     lrl1 = LR(penalty = "l1", solver = "liblinear" , C = i, max_iter = 1000)
#     lrl2 = LR(penalty = "l2", solver = "liblinear" , C = i, max_iter = 1000)
    
#     lrl1 = lrl1.fit(xtrain, ytrain)   
#     l1.append(accuracy_score(lrl1.predict(xtrain), ytrain))
#     l1test.append(accuracy_score(lrl1.predict(xtest), ytest))    
    
#     lrl2 = lrl2.fit(xtrain, ytrain)   
#     l2.append(accuracy_score(lrl2.predict(xtrain), ytrain))
#     l2test.append(accuracy_score(lrl2.predict(xtest), ytest))
    
# graph = [l1, l2, l1test, l2test]
# color = ["green","black", "lightgreen", "gray"]
# label = ["l1", "l2", "l1test" ,"l2test"]
# plt.figure(figsize = (6, 6)) 
# for i in range(len(graph)):
#     plt.plot(np.linspace(0.05, 1, 19), graph[i], color[i], label = label[i])
# plt.legend(loc = 4)
# plt.show()

