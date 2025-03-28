#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:24:39 2023

@author: user
"""
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression

def cross_validation2(k,xtrain,ytrain):
    kf = KFold(n_splits=k)
    
    x_lst_train = []
    x_lst_val = []
    y_lst_train = []
    y_lst_val = []
    
    for train_index, val_index in kf.split(xtrain):
        x_train, x_val = [], []
        y_train, y_val = [], []
        
        for index in train_index:
            x_train.append(xtrain[index])
            y_train.append(ytrain[index])
        
        for index in val_index:
            x_val.append(xtrain[index])
            y_val.append(ytrain[index])
        
        x_lst_train.append(x_train)
        x_lst_val.append(x_val)
        y_lst_train.append(y_train)
        y_lst_val.append(y_val)
        
    return x_lst_train, x_lst_val, y_lst_train, y_lst_val


#%%

#%%
def cross_validation(k,xtrain,ytrain):
    kf = KFold(n_splits=k)    

    x_lst_train=[]
    x_lst_val=[]
    y_lst_train=[]
    y_lst_val=[]
     
    x_train_index_list = []
    x_val_index_list = []
            
    for train_index, val_index in kf.split(xtrain):
        x_train_index_list.append(train_index)
        x_val_index_list.append(val_index)
        
    for kf in x_train_index_list:
        lst_v= [] 
        y_lst_v=[]
        for i in kf:
            lst_v.append(xtrain[i])
            y_lst_v.append(ytrain[i])
    
        
        x_lst_train.append(lst_v)
        y_lst_train.append(y_lst_v)
    
    for kf in x_val_index_list:
        lst_v= [] 
        y_lst_v = []
        for i in kf:
            lst_v.append(xtrain[i])
            y_lst_v.append(ytrain[i])
    
        x_lst_val.append(lst_v)
        y_lst_val.append(y_lst_v)
        
    return x_lst_train, x_lst_val, y_lst_train, y_lst_val



#%%
def nested_cross_validation(k_outer,k_inner,xtrain,ytrain):
    kf = KFold(n_splits=k_outer)    

    x_lst_train=[]
    x_lst_val=[]
    y_lst_train=[]
    y_lst_val=[]
     
    x_train_index_list = []
    x_val_index_list = []
            
    for train_index, val_index in kf.split(xtrain):
        x_train_index_list.append(train_index)
        x_val_index_list.append(val_index)
        
    for kf in x_train_index_list:
        lst_v= [] 
        y_lst_v=[]
        for i in kf:
            lst_v.append(xtrain[i])
            y_lst_v.append(ytrain[i])
    
        
        x_lst_train.append(lst_v)
        y_lst_train.append(y_lst_v)
    
    for kf in x_val_index_list:
        lst_v= [] 
        y_lst_v = []
        for i in kf:
            lst_v.append(xtrain[i])
            y_lst_v.append(ytrain[i])
    
        x_lst_val.append(lst_v)
        y_lst_val.append(y_lst_v)
        
    
    x_train_outer_fold,x_val_outer_fold,y_train_outer_fold,y_val_outer_fold=cross_validation(k_inner,x_lst_train,y_lst_train)

    x_train_inner_fold=[]
    x_val_inner_fold=[]
    y_train_inner_fold=[]
    y_val_inner_fold=[]



    for cv_x_train, cv_y_train in (zip(x_train_outer_fold,y_train_outer_fold)):
        x_train_inner_fold_var,x_val_inner_fold_var,y_train_inner_fold_var,y_val_inner_fold_var=cross_validation(k_outer,cv_x_train,cv_y_train)
            
        x_train_inner_fold.append(x_train_inner_fold_var)
        x_val_inner_fold.append(x_val_inner_fold_var)
        y_train_inner_fold.append(y_train_inner_fold_var)
        y_val_inner_fold.append(y_val_inner_fold)

    return x_lst_train, x_lst_val, y_lst_train, y_lst_val,x_train_inner_fold,x_val_inner_fold,y_train_inner_fold,y_val_inner_fold
