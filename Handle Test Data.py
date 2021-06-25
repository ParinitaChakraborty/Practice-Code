#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


test_df = pd.read_csv('/Users/parinita/Documents/Kaggle/house-prices-advanced-regression-techniques/test.csv')


# In[4]:


test_df.shape


# In[5]:


test_df.head()


# In[6]:


test_df.isnull().sum()


# In[7]:


test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())


# In[8]:


test_df['MSZoning']=test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])


# In[9]:


test_df.shape


# In[10]:


test_df.drop(['Alley'],axis=1,inplace=True)


# In[11]:


test_df.shape


# In[12]:


test_df['BsmtCond']=test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])
test_df['BsmtQual']=test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])


# In[13]:


test_df['FireplaceQu']=test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])
test_df['GarageType']=test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])


# In[14]:


test_df.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[15]:


test_df.shape


# In[16]:


test_df['GarageFinish']=test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])
test_df['GarageQual']=test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])
test_df['GarageCond']=test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])
test_df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[17]:


test_df.shape


# In[18]:


test_df.drop(['Id'],axis=1,inplace=True)


# In[19]:


test_df['MasVnrType']=test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])
test_df['MasVnrArea']=test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mode()[0])


# In[20]:


sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[21]:


test_df['BsmtExposure']=test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])


# In[23]:


sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[24]:


test_df['BsmtFinType2']=test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])


# In[25]:


test_df.loc[:,test_df.isnull().any()].head()


# In[26]:


test_df['Utilities']=test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])
test_df['Exterior1st']=test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])
test_df['Exterior2nd']=test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])
test_df['BsmtFinType1']=test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])
test_df['BsmtFinSF1']=test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())
test_df['BsmtFinSF2']=test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean())
test_df['BsmtUnfSF']=test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())
test_df['TotalBsmtSF']=test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())
test_df['BsmtFullBath']=test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])
test_df['BsmtHalfBath']=test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])
test_df['KitchenQual']=test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])
test_df['Functional']=test_df['Functional'].fillna(test_df['Functional'].mode()[0])
test_df['GarageCars']=test_df['GarageCars'].fillna(test_df['GarageCars'].mean())
test_df['GarageArea']=test_df['GarageArea'].fillna(test_df['GarageArea'].mean())
test_df['SaleType']=test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])


# In[27]:


test_df.shape


# In[28]:


test_df.to_csv('formulatedtest.csv',index=False)


# In[ ]:




