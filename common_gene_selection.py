# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 10:51:49 2022

@author: xia shufan
"""
import pandas 
import numpy as np
import scipy #  使用tf1 kernel 试一试
from scipy import stats
from numpy import median
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras import optimizers
from sklearn.metrics import accuracy_score # 这一次 sklearn 除了问题, !python -m pip install scikit-learn
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier # 不知道为什么浏览器总在转圈
from pathlib import Path
## 1. find file path 
data_prefix = Path(r"D:\project项目\MoGCN-master\rawMetabricData")
model_path = "" # model 没搞清楚是什么
# 接下来的数据处理从https://github.com/dmitrykazhdan/METABRIC-Autoencoder 参考
# mRMR 算法从https://github.com/smazzanti/mrmr 下载,!pip install mrmr_selection # 从执行上看好像成功了

import mrmr
cna_data_path = data_prefix.joinpath("raw_CNA_1.csv")
expr_data_path = data_prefix.joinpath("raw_gene_expression_1.csv")
patient_data_path =  data_prefix.joinpath("raw_clinical_1.csv")
model_name = model_path + "my_model"

## 2. define PAM50 genes and functions

# PAM50_genes = ['FOXC1', 'MIA', 'KNTC2', 'CEP55', 'ANLN',
#                'MELK', 'GPR160', 'TMEM45B',
#                'ESR1', 'FOXA1', 'ERBB2', 'GRB7',
#                'FGFR4', 'BLVRA', 'BAG1', 'CDC20',
#                'CCNE1', 'ACTR3B', 'MYC', 'SFRP1',
#                'KRT17', 'KRT5', 'MLPH', 'CCNB1', 'CDC6',
#                'TYMS', 'UBE2T', 'RRM2', 'MMP11',
#                'CXXC5', 'ORC6L', 'MDM2', 'KIF2C', 'PGR',
#                'MKI67', 'BCL2', 'EGFR', 'PHGDH',
#                'CDH3', 'NAT1', 'SLC39A6',
#                'MAPT', 'UBE2C', 'PTTG1', 'EXO1', 'CENPF',
#                'CDCA1', 'MYBL2', 'BIRC5'] # 
# Compute entropy for CNA variables
def entropy(x):
    unique, counts = np.unique(x, return_counts=True)
    counts = counts / sum(counts)
    return scipy.stats.entropy(counts)

# Compute Median Absolute Deviation for RNA variables
def MAD(x):
    return median(abs(x - median(x)))
# Compute Median Absolute Deviation for RNA variables

# Group CNA variables, that is not necessary ?
def normalize_cna(x):
    if x == -1 or x == -2:
        x = -1
    elif x == 1 or x == 2:
        x = 1
    else:
        x = 0
    return x

def transferLabelFromSurvivalData(data):
    pass

def train_graph(): 
    # Load patient data from file
    
    patient_data = pandas.read_csv(patient_data_path)
    survival_months_data = patient_data[['PATIENT_ID', 'OS_MONTHS']].dropna()
     # Load CNA data from file
    cna_data = pandas.read_csv(cna_data_path).dropna()
    cna_data = cna_data.drop(['Entrez_Gene_Id'], axis=1)# delete entrez column
    # Load gene expr data from file
    gene_expr_data = pandas.read_csv(expr_data_path).dropna()
    gene_expr_data = gene_expr_data.drop(['Entrez_Gene_Id'], axis=1)
    # Extract common genes
    common_genes = set(cna_data['Hugo_Symbol']) & set(gene_expr_data['Hugo_Symbol'])
    # common_with_PAM50 = common_genes & set(PAM50_genes)
    common_genes = pandas.Series(list(common_genes)).dropna() # previous selection
    cna_data = cna_data.loc[cna_data['Hugo_Symbol'].isin(common_genes)]
    gene_expr_data = gene_expr_data.loc[gene_expr_data['Hugo_Symbol'].isin(common_genes)]
    # Extract common patients
    common_cols = cna_data.columns.intersection(gene_expr_data.columns)
    cna_data = cna_data[common_cols]
    gene_expr_data= gene_expr_data[common_cols]

    # Sort by gene
    cna_data = cna_data.sort_values(by='Hugo_Symbol')
    gene_expr_data = gene_expr_data.sort_values(by='Hugo_Symbol')

    ## star: the Median Absolute Deviation (MAD) score, across all patients was computed for every gene
    
    # Extract most high-varied genes
    np_gene_data = gene_expr_data.iloc[:, 1:].values
    # 
    top_MAD_cna = np.argsort(np.apply_along_axis(func1d=MAD, axis=1, arr=np_gene_data))[-1200:]
    
    # Obtain list of genes to extract
    selected_genes = cna_data.iloc[top_MAD_cna, 0] # get one column 
    selected_genes = list(set(selected_genes) | common_genes)
    selected_genes = pandas.Series(list(selected_genes)).dropna()
    ## and update gene_expr_data and cna_data
    gene_expr_data = gene_expr_data.loc[gene_expr_data['Hugo_Symbol'].isin(selected_genes)]
    cna_data = cna_data.loc[cna_data['Hugo_Symbol'].isin(selected_genes)]
    # extract most entropy gene for top 300 gene
    np_gene_data = cna_data.iloc[:, 1:].values
    top_MAD_cna = np.argsort(np.apply_along_axis(func1d=entropy, axis=1, arr=np_gene_data))[-300:]
    ## update cna_data
    selected_genes = cna_data.iloc[top_MAD_cna, 0]
    cna_data = cna_data.loc[cna_data['Hugo_Symbol'].isin(selected_genes)]
    
    # Convert CNA to one-hot encoding
    cna_data = cna_data.iloc[:, 1:]
    cna_data = cna_data.applymap(normalize_cna)# 
    cna_data = cna_data.transpose()
    print(cna_data)
    cna_data = pandas.get_dummies(cna_data, columns=cna_data.columns)
    cna_data = cna_data.transpose()
    
    # Remove gene column from RNA, named gene_expr_data
    gene_expr_data = gene_expr_data.iloc[:, 1:]
    # Get number of features
    n_cna_features = cna_data.shape[0]
    n_gene_expr_features = gene_expr_data.shape[0]
    print("CNA features: ", n_cna_features)# 900
    print("RNA features: ", n_gene_expr_features)# 15709
    
    # uptil 129 ,we have 28 row codes
    
    np_type_data = []
    np_gene_expr_data = []
    np_cna_data = []
    
    for index, row in survival_months_data.iterrows():

        patient_id = row['PATIENT_ID']
        survival_id = row['OS_MONTHS']
        
        if survival_id > 60 : #  survival months more than 60 ,label as long terminal
            survival_id = 1
        else:       
            survival_id = 0
        if patient_id in gene_expr_data: 
            #  I don't have to limit num of each survival_id
            gene_expr_sample = gene_expr_data[patient_id].values.transpose()
            cna_sample = cna_data[patient_id].values.transpose()
            np_gene_expr_data.append(gene_expr_sample)
            np_cna_data.append(cna_sample)
            np_type_data.append(survival_id)
            
    np_gene_expr_data = np.array(np_gene_expr_data) # negative numbers
    np_cna_data = np.array(np_cna_data)
    np_type_data = np.array(np_type_data)
    
    # Normalize gene expr data
    np_gene_expr_data = 2 * (np_gene_expr_data - np.min(np_gene_expr_data)) / (np.max(np_gene_expr_data) - np.min(np_gene_expr_data)) - 1

    # Print cluster counts
    unique, counts = np.unique(np_type_data, return_counts=True)
    print(counts)
    
    # Split into training and test data
    n_samples = np_gene_expr_data.shape[0]
    n_train_samples = int(n_samples * 0.8)
    sample_indices = np.arange(n_samples)
    np.random.shuffle(sample_indices)
    train_indices = sample_indices[:n_train_samples]
    test_indices = sample_indices[n_train_samples:]
    
    X_train_gene_expr = np_gene_expr_data[train_indices, :].copy()
    X_train_cna = np_cna_data[train_indices, :].copy()
    y_train = np_type_data[train_indices].copy()

    X_test_gene_expr = np_gene_expr_data[test_indices, :].copy()
    X_test_cna = np_cna_data[test_indices, :].copy()
    y_test = np_type_data[test_indices].copy()
    
train_graph()

                    



    