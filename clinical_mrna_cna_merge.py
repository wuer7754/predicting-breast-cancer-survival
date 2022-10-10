# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:23:12 2022

@author: xia shufan
"""

import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
from sklearn.preprocessing import KBinsDiscretizer
data_prefix = Path(r"D:\project项目\MoGCN-master\rawMetabricData")
data_path =  data_prefix.joinpath("processed_clinical.csv")

data = pd.read_csv(data_path,index_col = 0).dropna()
data = data.drop("SEX",axis = 1)   
data = data.drop("VITAL_STATUS",axis = 1)  
data = data[data.index.str.startswith("MB")]

################################  补充缺失值： impleImputer #######################################
data.isnull().sum()

from sklearn.impute import SimpleImputer
age = data_.loc[:,"AGE_AT_DIAGNOSIS"].values.reshape(-1,1)
data.loc[:,"AGE_AT_DIAGNOSIS"] = SimpleImputer(strategy ="median").fit_transform(age)

ly_node = data.loc[:,"LYMPH_NODES_EXAMINED_POSITIVE"].values.reshape(-1,1)
data.loc[:,"LYMPH_NODES_EXAMINED_POSITIVE"] = SimpleImputer().fit_transform(ly_node)

thgene = data.loc[:,"THREEGENE"].values.reshape(-1,1)
data.loc[:,"THREEGENE"] = SimpleImputer(strategy = "median").fit_transform(thgene)

LATERALITY  =  data_.loc[:,"LATERALITY"].values.reshape(-1,1)
data.loc[:,"LATERALITY"] = SimpleImputer(strategy = "most_frequent").fit_transform(LATERALITY)

HISTOLOGICAL_SUBTYPE  = data.loc[:,"LATERALITY"].values.reshape(-1,1)
data.loc[:,"HISTOLOGICAL_SUBTYPE"] = SimpleImputer(strategy = "most_frequent").fit_transform(LATERALITY)

OS_MONTHS = data.loc[:,"OS_MONTHS"].values.reshape(-1,1)
data.loc[:,"OS_MONTHS"] = SimpleImputer(strategy = "median").fit_transform(OS_MONTHS)

OS_STATUS = data.loc[:,"OS_STATUS"].values.reshape(-1,1)
data.loc[:,"OS_STATUS"] = SimpleImputer(strategy = "median").fit_transform(OS_STATUS)

BREAST_SURGERY = data.loc[:,"BREAST_SURGERY"].values.reshape(-1,1)
data.loc[:,"BREAST_SURGERY"] = SimpleImputer(strategy = "most_frequent").fit_transform(BREAST_SURGERY)

ER_IHC = data.loc[:,"ER_IHC"].values.reshape(-1,1)
data.loc[:,"ER_IHC"] = SimpleImputer(strategy = "median").fit_transform(ER_IHC)

HER2_SNP6  = data.loc[:,"HER2_SNP6"].values.reshape(-1,1)
data.loc[:,"HER2_SNP6"] = SimpleImputer(strategy = "median").fit_transform(HER2_SNP6)

RADIO_THERAPY  = data.loc[:,"RADIO_THERAPY"].values.reshape(-1,1)
data.loc[:,"RADIO_THERAPY"] = SimpleImputer(strategy = "most_frequent").fit_transform(BREAST_SURGERY)

data_.isnull().sum()
THREEGENE  = data_.loc[:,"THREEGENE"].values.reshape(-1,1)
data.loc[:,"THREEGENE"] = SimpleImputer(strategy = "most_frequent").fit_transform(THREEGENE )

LYMPH_NODES_EXAMINED_POSITIVE = data_.loc[:,"LYMPH_NODES_EXAMINED_POSITIVE"].values.reshape(-1,1)
data.loc[:,"LYMPH_NODES_EXAMINED_POSITIVE"] = SimpleImputer(strategy = "most_frequent").fit_transform(LYMPH_NODES_EXAMINED_POSITIVE )
######################## 将连续值分段 ： KBinsDiscretizer()####################################
X =  data.loc[:,"OS_MONTHS"].values.reshape(-1,1)     
bins_ = KBinsDiscretizer(n_bins = 5 , encode = "ordinal",strategy = "uniform")
data.loc[:,"OS_MONTHS"] =  bins_.fit_transform(X)


X =  data_.loc[:,"OS_MONTHS"].values.reshape(-1,1) 
bins_ = KBinsDiscretizer(n_bins = 5 , encode = "ordinal",strategy = "uniform")
data.loc[:,"OS_MONTHS"] =  bins_.fit_transform(X)

data.to_csv("processed_clinical.csv")

########################### merge data processed ###############################

data_prefix = Path(r"D:\project项目\MoGCN-master\reductionMetabricData")
patient_data_path =  data_prefix.joinpath("Breast_patient_data.csv")
# data = pd.read_csv(data_path,index_col = 0)
# data = data.drop("SEX",axis = 1)   
# data = data.drop("VITAL_STATUS",axis = 1)  
# data = data[data.index.str.startswith("MB")]

cna_data_path = data_prefix.joinpath("reduction_cna.csv")
expr_data_path = data_prefix.joinpath("reduction_gene_expr.csv")
rna_data_path = data_prefix.joinpath("reduction_rna.csv")
type_data_path = data_prefix.joinpath("type_data.csv")

def feature_selection(cna_data_path, expr_data_path, rna_data_path, patient_data_path):
    # Load patient data from file    
    patient_data = pd.read_csv(patient_data_path, header = 0, index_col = 0)
    patient_data.isnull().sum()
    
     # Load CNA data from file
    cna_data = pd.read_csv(cna_data_path,header = 0, index_col = 0).dropna()
    cna_data = cna_data.drop(['Entrez_Gene_Id'], axis=1)# delete entrez column
    # Load gene expr data from file
    gene_expr_data = pd.read_csv(expr_data_path,header = 0, index_col = 0).dropna()
    gene_expr_data = gene_expr_data.drop(['Entrez_Gene_Id'], axis=1)
    # Load mRNA data from file
    rna_data = pd.read_csv(rna_data_path, header = 0, index_col = 0).dropna()
    rna_data = rna_data.drop(['Entrez_Gene_Id'], axis=1)
    
    type_data = pd.read_csv(type_data_path, header = None) # 读这种没有行名的,header = None 添加 一行 index 0 1 2
    
    #data = pd.concat([data])
    # cols are  patient_id
    common_cols = cna_data.columns.intersection(gene_expr_data.columns)
    common_cols = common_cols.intersection(rna_data.columns)
    cna_data = cna_data[common_cols]  # till now , (18847,1905)
    gene_expr_data= gene_expr_data[common_cols] # (18484,1905)
    rna_data = rna_data[common_cols] # same with the other (18848,1905)
    patient_data = patient_data.loc[patient_data['PATIENT_ID'].isin(common_cols)] # (1904, 22)
    survival_months_data = patient_data[['PATIENT_ID', 'OS_MONTHS']].dropna()
    
    np_type_data = []
    np_gene_expr_data = []
    np_cna_data = []
    np_rna_data = []
    np_patient_data = []
    
    for index, row in survival_months_data.iterrows():
      
        patient_id = row['PATIENT_ID']
        survival_id = row['OS_MONTHS']           
        if patient_id in gene_expr_data: 
            #  I don't have to limit num of each survival_id
            gene_expr_sample = gene_expr_data[patient_id].values.transpose()
            cna_sample = cna_data[patient_id].values.transpose()
            rna_sample = rna_data[patient_id].values.transpose()
            patient_sample = patient_data.loc[patient_data['PATIENT_ID'] == patient_id]
            patient_sample = patient_sample.iloc[0,1:].values.astype('int64')
            np_gene_expr_data.append(gene_expr_sample)
            np_cna_data.append(cna_sample)
            np_rna_data.append(rna_sample)
            np_patient_data.append(patient_sample)
            np_type_data.append(survival_id)
            
    np_gene_expr_data = np.array(np_gene_expr_data) # negative numbers
    np_cna_data = np.array(np_cna_data)
    np_rna_data = np.array(np_rna_data)
    np_patient_data = np.array(np_patient_data)
    np_type_data = np.array(np_type_data)
    # 连接数组arr 包括np_patient_data，np_cna_data，np_rna_data
    arr = np.concatenate([np_patient_data,np_cna_data],axis = 1)
    arr = np.concatenate([arr, np_rna_data], axis = 1)
    
    arr = np.concatenate([patient_data, cna_data],axis = 1)
    arr = np.concatenate([arr, gene_expr_data], axis = 1)
    
    
    from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline 
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report,confusion_matrix
    from sklearn.model_selection import GridSearchCV
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier # 在pca上 结果不是很好
    from sklearn.manifold import  TSNE, LocallyLinearEmbedding
    from sklearn.feature_selection import SelectFromModel
    
    X = arr
    y = type_data
    sc = StandardScaler()
    sc.fit(X)
    
    ############### pca: Accuracy :  0.5231138054560485 ############
   
   #  Classification Report : 
   #             precision    recall  f1-score   support

   #       0.0       0.63      0.53      0.58      1242
   #       4.0       0.32      0.42      0.36       662

   #  accuracy                           0.49      1904
   # macro avg       0.48      0.47      0.47      1904
   # weighted avg       0.52      0.49      0.50      1904
   
    lda = PCA(n_components = 100) # 
    lr = LogisticRegression()  
    X_reduced = lda.fit_transform(X)
   
    # pipeline = Pipeline(steps=[('lda',lda),('logistic',lr)])  
    pipeline = Pipeline(steps=[('logistic',lr)])  
    pipeline.fit(X, y)
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)
    results = cross_val_score(pipeline, X, y, scoring='accuracy', cv=kfold)
    predict = cross_val_predict(pipeline, X,y,cv=10)
    print("Accuracy : ",results.mean())  
    print("Classification Report : \n",classification_report(predict,y))
    print("Confusion Metrix : \n",confusion_matrix(predict,y))
 #####################  logisticRegression ############################:
    # Accuracy :  0.9658638743455498
    # Classification Report : 
    #            precision    recall  f1-score   support

    #      0.0       1.00      1.00      1.00       560
    #      1.0       0.99      1.00      1.00       599
    #      2.0       0.98      0.96      0.97       449
    #      3.0       0.91      0.88      0.90       284
    #      4.0       0.07      0.17      0.10        12
    lr = LogisticRegression()    
    # pipeline = Pipeline(steps=[('lda',lda),('logistic',lr)])  
    pipeline = Pipeline(steps=[('logistic',lr)])  
    pipeline.fit(X, y)
    kfold = KFold(n_splits=10, random_state=1, shuffle=True)
    
    ### 重新划分train,test,val
    results = cross_val_score(pipeline, X, y, scoring='accuracy', cv=kfold)
    predict = cross_val_predict(pipeline, X,y,cv=10)
    print("Accuracy : ",results.mean())  
    print("Classification Report : \n",classification_report(predict,y))
    print("Confusion Metrix : \n",confusion_matrix(predict,y)) 
    
####################### lle : Accuracy :  0.5477790024800221#########
    
   #   precision    recall  f1-score   support

   #       0.0       1.00      0.55      0.71      1904
   #       4.0       0.00      0.00      0.00         0

   #  accuracy                           0.55      1904
   # macro avg       0.50      0.27      0.35      1904
   # weighted avg       1.00      0.55      0.71      1904
    lle = LocallyLinearEmbedding(n_components = 3,n_neighbors = 8)
    X = lle.fit_transform(X)
    pipeline = Pipeline(steps=[('lle',lle),('logistic',lr)])     
    pipeline.fit(X, y)
    kfold = KFold(n_splits=10, random_state=1,shuffle=True)
    results = cross_val_score(pipeline, X, y, scoring='accuracy', cv=kfold)
    predict = cross_val_predict(pipeline,X,y,cv=10)
    print("Accuracy : ",results.mean())  
    print("Classification Report : \n",classification_report(predict,y))
    print("Confusion Metrix : \n",confusion_matrix(predict,y))
    
#################### mrmr 
    import  mrmr
    from mrmr import mrmr_classif
    X = patient_data.iloc[:,1:]
    X = X.drop("OS_MONTHS",axis = 1)  
    y = pd.Series(np_type_data)
    selected_features = mrmr_classif(X= X, y=y, K=10)
