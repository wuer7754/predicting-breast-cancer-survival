# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 18:25:21 2022

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

# 接下来的数据处理从https://github.com/dmitrykazhdan/METABRIC-Autoencoder 参考
# mRMR 算法从https://github.com/smazzanti/mrmr 下载,!pip install mrmr_selection # 从执行上看好像成功了

# import mrmr

data_prefix = Path(r"D:\project项目\MoGCN-master\rawMetabricData")
model_path = "" # model 没搞清楚是什么
cna_data_path = data_prefix.joinpath("raw_CNA_1.csv")
expr_data_path = data_prefix.joinpath("raw_gene_expression_1.csv")
patient_data_path =  data_prefix.joinpath("raw_clinical_1.csv")
rna_data_path = data_prefix.joinpath("raw_mRNA_1.csv")
model_name = model_path + "my_model"

def normalize_cna(x):
    if x == -1 or x == -2:
        x = -1
    elif x == 1 or x == 2:
        x = 1
    else:
        x = 0
    return x


# 因子分析， 返回降维后的新数据
def factoryAnalysis(data): # index_column 行名 str 提取所有
    df = data.loc[:,1:] #  传进来的值是 dataframe
    fa = FactorAnalysis(n_components = 400) # 指定400 个因子 作为新变量
    fa.fit(df)
    trans_x = fa.transform(df)
    factor_column = []
    for index in range(400):
        tmp = 'factor' + str(index + 1)
        factor_column.append(tmp)
    tran_df = pandas.DataFrame(trans_x, column = factor_column)
    return tran_df

    
def feature_reduction():
    ######################## 1.  load data ###############################
    # Load patient data from file    
    patient_data = pandas.read_csv(patient_data_path)
    survival_months_data = patient_data[['PATIENT_ID', 'OS_MONTHS']].dropna()
     # Load CNA data from file
    cna_data = pandas.read_csv(cna_data_path).dropna()
    cna_data = cna_data.drop(['Entrez_Gene_Id'], axis=1)# delete entrez column
    # Load gene expr data from file
    gene_expr_data = pandas.read_csv(expr_data_path).dropna()
    gene_expr_data = gene_expr_data.drop(['Entrez_Gene_Id'], axis=1)
    # Load mRNA data from file
    rna_data = pandas.read_csv(rna_data_path).dropna()
    rna_data = rna_data.drop(['Entrez_Gene_Id'], axis=1)
    
    
    ###################### 2. 对齐数据集 #############################
    # Extract common genes, num of common genes is 15709
    common_genes = set(cna_data['Hugo_Symbol']) & set(gene_expr_data['Hugo_Symbol'])
    common_genes = common_genes & set(rna_data['Hugo_Symbol']) # try 
    common_genes = pandas.Series(list(common_genes)).dropna() # previous selection
    cna_data = cna_data.loc[cna_data['Hugo_Symbol'].isin(common_genes)] 
    gene_expr_data = gene_expr_data.loc[gene_expr_data['Hugo_Symbol'].isin(common_genes)]
    rna_data =rna_data.loc[rna_data['Hugo_Symbol'].isin(common_genes)]
    # Extract common patients, num of common column is 1905
    common_cols = cna_data.columns.intersection(gene_expr_data.columns)
    common_cols = common_cols.intersection(rna_data.columns)
    cna_data = cna_data[common_cols]  # till now , (15709,1905)
    gene_expr_data= gene_expr_data[common_cols] # same with cna_data
    rna_data = rna_data[common_cols] # same with the other (15709,1905)
    
    
    ##################### 3. Sort by gene ################################
    cna_data = cna_data.sort_values(by='Hugo_Symbol')
    gene_expr_data = gene_expr_data.sort_values(by='Hugo_Symbol')
    rna_data = rna_data.sort_values(by='Hugo_Symbol')
    cna_data = cna_data.sort_index(axis = 0)
    gene_expr_data = gene_expr_data.sort_index(axis = 0)
    rna_data = rna_data.sort_index(axis = 0)
    ##################### 4. 制作多分类标签 ################################
    '''
    1.rmrm   需要 y
    2. factory analysis 暂时不会写
    '''
    import statistics
    survival_id =  survival_months_data['OS_MONTHS']
    median_survival = median(survival_id)
    mean_survival  = statistics.mean(survival_id)
    print(median_survival) # 116.4666
    print(mean_survival) #125.24427057011307
    
    
    np_type_data = []
    np_patient_id = []
   
    # global df_gene_expr_data
    # global df_cna_data 
    # global df_rna_data
    # gene_columns = gene_expr_data.columns.tolist()
    # cna_columns = cna_data.columns.tolist()
    # rna_columns = rna_data.columns.tolist()
    # df_gene_expr_data = pandas.DataFrame(columns = gene_columns )
    # df_cna_data = pandas.DataFrame(columns = cna_columns)
    # df_rna_data = pandas.DataFrame(columns = rna_columns)
    # df_type_data = pandas.DataFrame()
   
    for index, row in survival_months_data.iterrows():
        
        patient_id = row['PATIENT_ID']
        survival_id = row['OS_MONTHS'] 
        
        if survival_id > 60 and survival_id < median_survival: #  survival months more than 60 ,label as long terminal
            survival_id = 1
        elif survival_id  > median_survival :       
            survival_id = 2
        else :
            survival_id = 0
        if patient_id in gene_expr_data.columns: 
            # value 是什么 ？cna_data.iloc[1].values结果 是array 数组 array(['A1CF', 0, 0, ..., 0.0, 0.0, 0.0], dtype=object)
            np_patient_id.append(patient_id)
            np_type_data.append(survival_id)
            # gene_expr_sample = gene_expr_data[patient_id] # 加不加transpose()好像没区别,cna_data['MB-0000'].transpose()结果是Name: MB-0000, Length: 15709, dtype: int64
            # cna_sample = cna_data[patient_id]
            # rna_sample = rna_data[patient_id]      
            
            # 使用loc 函数增添一行， index 可以是变量patient_id
            # 错误 Must have equal len keys and value when setting with an iterable
            # 这种方法虽然能运行但是太慢了， loc函数每次都要查找
            # df_gene_expr_data.loc[patient_id] = gene_expr_sample
            # df_cna_data.loc[patient_id] = cna_sample
            # df_rna_data.loc[patient_id] = rna_sample
            # np_type_data.iloc[patient_id] = survival_id           
            # 或者使用concat函数
            # pandas.concat([df_gene_expr_data, gene_expr_sample], axis = 1)
            # pandas.concat([df_cna_data, cna_sample], axis = 1)
            # pandas.concat([df_rna_data, rna_sample], axis = 1)
            # np_type_data.append(survival_id)
            # pandas.concat([df_type_data,])
    df_survival = pandas.DataFrame(data= np_type_data, index = np_patient_id ) # 1904,1 [473 487 944]
    # Convert CNA to one-hot encoding
    cna_data = cna_data.iloc[:, 1:]
    cna_data = cna_data.applymap(normalize_cna)# 
    cna_data = cna_data.transpose()
    print(cna_data)
    cna_data = pandas.get_dummies(cna_data, columns=cna_data.columns)
    cna_data = cna_data.transpose()
    print(cna_data)
    ###########################      
    cna_t = cna_data.T
    rna_t = rna_data.T
    gene_t = gene_expr_data.T
    cna_t = cna_t.sort_values(by='Index',axis = 1, ascending =True) # 为什么不行,
    cna_columns = cna_t.iloc[0,:].tolist()
    cna_t.cloumns = cna_columns
    cna_t = cna_t.drop(labels= 'class',axis = 1)
    cna_t.insert(loc = 0 ,column = 'class', value = df_survival[0]) 
   
    unique, counts = np.unique(df_survival, return_counts=True)
    print(counts)
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
    ##################  pca (100) + lle (3) 0.495 ###############
    X = cna_t.iloc[1:,1:].values
    y = cna_t.iloc[1:,0].values
    sc = StandardScaler()
    sc.fit(X)
    lda = PCA(n_components = 100) # 不能转化为 4维以上的数据
    lr = LogisticRegression()
    # pipeline 是什么意思 ？
    
    X_reduced = lda.fit_transform(X)
    lle = LocallyLinearEmbedding(n_components = 3,n_neighbors = 8)
    X = lle.fit_transform(X_reduced)
    pipeline = Pipeline(steps=[('lda',lle),('logistic',lr)])     
    pipeline.fit(X, y)
    kfold = KFold(n_splits=10, random_state=1,shuffle=True)
    results = cross_val_score(pipeline, X, y, scoring='accuracy', cv=kfold)
    predict = cross_val_predict(pipeline,X,y,cv=10)
    print("Accuracy : ",results.mean())  
    print("Classification Report : \n",classification_report(predict,y))
    print("Confusion Metrix : \n",confusion_matrix(predict,y))
    
    ##################  tsne  0.468 ########################
    from sklearn.model_selection import GridSearchCV
    X = rna_t.iloc[1:,1:].values
    y = rna_t.iloc[1:,0].values
    sc = StandardScaler()
    sc.fit(X)
    
    tsne = TSNE(n_components=3, init='pca', random_state=0)
    X = tsne.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=444)
    
    sel = SelectFromModel(ExtraTreesClassifier(n_estimators=10, random_state=444), threshold='mean')
    clf = RandomForestClassifier(n_estimators= 500, random_state=444)
    model = Pipeline([('sel', sel), ('clf', clf)])
    params = {'clf__max_features': ['auto', 'sqrt', 'log2']}
    gs = GridSearchCV(model,params)
    gs.fit(X_train,y_train)
    
    gs.score(X_test, y_test)
    print("Classification Report : \n",classification_report(predict,y)) # predict 有问题         
    
    ########################## fa
    lr = LogisticRegression()
    # pipeline 是什么意思 ？
    from sklearn.decomposition import FactorAnalysis
    fa = FactorAnalysis(n_components = 400) # 指定400 个因子 作为新变量
    fa.fit(X)
    trans_x = fa.transform(df)
    model = Pipeline(steps=[('tsne',tsne),('logistic',lr)])     
    pipeline.fit(X, y)
    kfold = KFold(n_splits=10, random_state=1,shuffle=True)
    results = cross_val_score(pipeline, X, y, scoring='accuracy', cv=kfold)
    predict = cross_val_predict(pipeline,X,y,cv=10)
    print("Accuracy : ",results.mean())  # cna: 0.3934 (lda), 0.48 (pca), 
    print("Classification Report : \n",classification_report(predict,y))
    print("Confusion Metrix : \n",confusion_matrix(predict,y))
    ###
    ## ndarray 
    cna_data_1 = cna_data.iloc[:,1:]
    np_gene_data = gene_expr_data.iloc[:, 1:].values 
    np_rna_data = rna_data.iloc[:, 1:].values
    np_cna_data = cna_data.iloc[:, 1:].values
    
    



fa_400_gene = factoryAnalysis(gene_expr_data)