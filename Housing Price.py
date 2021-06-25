#!/usr/bin/env python
# coding: utf-8

# # Kaggle Competition for House Prices : Advanced Regression Techniques 

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('/Users/parinita/Documents/Kaggle/house-prices-advanced-regression-techniques/train.csv')


# In[3]:


df.head()


# In[4]:


#counting rows
df.shape


# In[5]:


#checking for null values
df.isnull().sum()


# In[6]:


#heatmap for null values
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[7]:


#information of the field
df.info()


# In[8]:


df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[9]:


df['MSZoning'].value_counts()


# In[10]:


df.drop(['Alley'],axis=1,inplace=True)


# In[11]:


df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])


# In[12]:


df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
#FireplaceQu
#GarageType


# In[13]:


df.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[14]:


df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
#GarageFinish
#GarageQual
#GarageCond


# In[15]:


df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[16]:


df.shape


# In[17]:


df.drop(['Id'],axis=1,inplace=True)


# In[18]:


df.isnull().sum()


# In[19]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[20]:


df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[21]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')


# In[22]:


df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])


# In[23]:


df.dropna(inplace=True)


# In[24]:


df.shape


# In[25]:


df.head()


# In[26]:


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']


# In[27]:


len(columns)


# In[28]:


def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
    df_final=pd.concat([final_df,df_final],axis=1)
    return df_final


# In[29]:


main_df=df.copy()


# In[30]:


##Combine test data
test_df=pd.read_csv('formulatedtest.csv')


# In[31]:


test_df.shape


# In[32]:


test_df.head()


# In[33]:


final_df=pd.concat([df,test_df],axis=0)


# In[34]:


final_df.shape


# In[35]:


final_df=category_onehot_multcols(columns)


# In[36]:


final_df.shape


# In[37]:


final_df=final_df.loc[:,~final_df.columns.duplicated()]


# In[38]:


final_df.shape


# In[39]:


final_df.head()


# In[40]:


df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]


# In[41]:


df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[42]:


df_Test.shape


# In[43]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# In[45]:


import xgboost
classifier=xgboost.XGBRegressor()
classifier.fit(X_train,y_train)


# In[47]:


from sklearn.ensemble import RandomForestRegressor


# In[48]:


import pickle
filename='finalized_model.pk1'
pickle.dump(classifier,open(filename,'wb'))


# In[49]:


y_pred=classifier.predict(df_Test)


# In[50]:


y_pred


# In[54]:


pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('/Users/parinita/Documents/Kaggle/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('/Users/parinita/Documents/Kaggle/house-prices-advanced-regression-techniques/sample_submission.csv',index=False)


# In[ ]:




