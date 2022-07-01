import numpy as np
import pandas as pd
import os
import random
import scipy.io as sio
from libsvm.svmutil import *
from pyHSICLasso import HSICLasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from numpy import sort
from xgboost import XGBClassifier

os.chdir('/Volumes/Volume/Downloads/')
fileNames=['Lung_Cancer','Kidney_Cancer','Glioma']
featureNum=[5,10,15,20,25,30]
times=10

for fileName in fileNames:
    data=pd.read_csv(fileName+'.csv')
    result=data['class']
    accValLasso=np.zeros((times,len(featureNum)),dtype='float')
    accValXGBoost=np.zeros((times,len(featureNum)),dtype='float')
    accValRandomForest=np.zeros((times,len(featureNum)),dtype='float')
    featureSelectedLasso=pd.DataFrame()
    featureSelectedXGBoost=pd.DataFrame()
    featureSelectedRandomForest=pd.DataFrame()

    #using libsvm for validation test
    def testBySvmlib(x_train,y_train,x_test,y_test,feats,j,i,funName):
        prob  = svm_problem(np.array(y_train),np.array(x_train[feats]))
        param = svm_parameter('-t 0 -c 4 -b 1')
        model = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(np.array(y_test), np.array(x_test[feats]), model)
        #print(p_label,p_acc,p_val)
        if(funName=='Lasso'):
            accValLasso[i][j]=p_acc[0]
        elif(funName=='XGBoost'):
            accValXGBoost[i][j]=p_acc[0]
        elif(funName=='RandomForest'):
            accValRandomForest[i][j]=p_acc[0]

    #algorithms 1: hsic_lasso
    def myLasso(data_train,data_test,j,i):
        X=data_train.iloc[:,1:]
        Y=data_train['class']
        Xt=data_test.iloc[:,1:]
        Yt=data_test['class']
        featsAll=X.columns.tolist()
        hsic_lasso = HSICLasso()
        hsic_lasso.input(np.array(X),np.array(Y),featname=featsAll)
        hsic_lasso.classification(featureNum[j])
        #hsic_lasso.dump()
        #hsic_lasso.save_param(filename='myLasso_featNum'+str(featureNum[j])+'_times'+str(i+1)+'.csv')
        feats=hsic_lasso.get_features()
        testBySvmlib(X,Y,Xt,Yt,feats,j,i,'Lasso')
        return feats

    #algorithms 2: xgboost
    def myXGBoost(data_train,data_test,j,i):
        X=data_train.iloc[:,1:]
        Y=data_train['class']
        Xt=data_test.iloc[:,1:]
        Yt=data_test['class']
        model = XGBClassifier()
        model.fit(np.array(X),np.array(Y))
        featsAll=X.columns.tolist()
        importances=model.feature_importances_
        index_imp=np.argsort(importances)[::-1]
        feats=[featsAll[f] for f in list(index_imp[0:featureNum[j]])]
        testBySvmlib(X,Y,Xt,Yt,feats,j,i,'XGBoost')
        return feats

    #algorithms 2: randomforest
    def myRandomForest(data_train,data_test,j,i):
        X=data_train.iloc[:,1:]
        Y=data_train['class']
        Xt=data_test.iloc[:,1:]
        Yt=data_test['class']
        randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
        model = randomforest.fit(np.array(X),np.array(Y))
        featsAll=X.columns.tolist()
        importances=model.feature_importances_
        index_imp=np.argsort(importances)[::-1]
        feats=[featsAll[f] for f in list(index_imp[0:featureNum[j]])]
        testBySvmlib(X,Y,Xt,Yt,feats,j,i,'RandomForest')
        return feats

    #run ten times for every number of selected features
    for j in range(len(featureNum)):
        featuresLasso=[]
        featuresXGBoost=[]
        featuresRandomForest=[]
        for i in range(times):
            #split the dataset into training dataset and testing dataset
            data_train,data_test=train_test_split(data,stratify=result,random_state=random.randint(1,100))
            #使用pyHSICLasso进行特征选择
            featsLasso = myLasso(data_train,data_test,j,i)
            featuresLasso.append(','.join(featsLasso))
            featsXGBoost=myXGBoost(data_train,data_test,j,i)
            featuresXGBoost.append(','.join(featsXGBoost))
            featsRandomForest=myRandomForest(data_train,data_test,j,i)
            featuresRandomForest.append(','.join(featsRandomForest))
        featureSelectedLasso['featNum'+str(featureNum[j])]=featuresLasso
        featureSelectedXGBoost['featNum'+str(featureNum[j])]=featuresXGBoost
        featureSelectedRandomForest['featNum'+str(featureNum[j])]=featuresRandomForest
    print('--------------final---------------')
    #for item in [accValLasso,accValXGBoost,accValRandomForest]:

    accMeanLasso = np.mean(accValLasso,0)
    accValAllLasso=np.vstack((accValLasso,accMeanLasso))
    accValAllLasso=pd.DataFrame(accValAllLasso)
    accValAllLasso.columns=['featNum'+str(i) for i in featureNum]
    accValAllLasso.index=['acc'+str(i) for i in range(1,times+1)]+['acc_mean']
    accValAllLasso.to_csv(fileName+'_accValAllLasso.csv')

    accMeanXGBoost = np.mean(accValXGBoost,0)
    accValAllXGBoost=np.vstack((accValXGBoost,accMeanXGBoost))
    accValAllXGBoost=pd.DataFrame(accValAllXGBoost)
    accValAllXGBoost.columns=['featNum'+str(i) for i in featureNum]
    accValAllXGBoost.index=['acc'+str(i) for i in range(1,times+1)]+['acc_mean']
    accValAllXGBoost.to_csv(fileName+'_accValAllXGBoost.csv')

    accMeanRandomForest = np.mean(accValRandomForest,0)
    accValAllRandomForest=np.vstack((accValRandomForest,accMeanRandomForest))
    accValAllRandomForest=pd.DataFrame(accValAllRandomForest)
    accValAllRandomForest.columns=['featNum'+str(i) for i in featureNum]
    accValAllRandomForest.index=['acc'+str(i) for i in range(1,times+1)]+['acc_mean']
    accValAllRandomForest.to_csv(fileName+'_accValAllRandomForest.csv')

    featureSelectedLasso.to_csv(fileName+'_featureSelectedLasso.csv')
    featureSelectedXGBoost.to_csv(fileName+'_featureSelectedXGBoost.csv')
    featureSelectedRandomForest.to_csv(fileName+'_featureSelectedRandomForest.csv')