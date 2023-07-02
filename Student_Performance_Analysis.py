#!/usr/bin/env python
# coding: utf-8

# In[1]:


from warnings import filterwarnings
filterwarnings('ignore')


# In[9]:


import os
os.chdir('D:/Datasets')


# In[11]:


import pandas as pd
df = pd.read_csv('Student_Performance.csv')


# In[12]:


df.head()


# In[13]:


df.info()


# In[14]:


df.columns


# In[15]:


df.shape


# In[16]:


df.describe()


# In[17]:


#seperate Categorical and continous features
df.dtypes


# In[20]:


def catconsep(df):
    cat = list(df.columns[df.dtypes=='object'])
    con = list(df.columns[df.dtypes!='object'])
    return cat,con


# In[21]:


cat,con = catconsep(df)
cat


# In[22]:


con


# ## Visualization using matplotlib and seaborn

# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[25]:


# univariate Analysis for continous features draw Histogram
for i in con:
    sns.histplot(data=df,x=i,kde=True)
    plt.title(f'Histogram of {i}')
    plt.show()
    


# In[27]:


# for continous features lets draw pie chart
df['Extracurricular Activities'].value_counts().plot(kind = 'pie')


# In[29]:


# correlation heatmap for continous Variables
sns.heatmap(df.corr(),annot= True, fmt ='.2f')


# ## Target Variable: Performance Index
# * Bivariate Analysis
# 1. cat vs con : Boxplot
# 2. cat vs cat : Crosstab
# 3. con vs con : Scatterplot

# In[34]:


#1.Cat Vs. Con : Drawing Boxplot for Performance Index vs extra curricular activities
sns.boxplot(data=df,x=df['Extracurricular Activities'], y= df['Performance Index'])
plt.xlabel = ('Extracurricular Activities')
plt.ylabel = ('Performance Index')
plt.title('Boxplot for Extracurricular Activities vs Performance Index')
plt.show()


# In[53]:


for i in con:
    if i!='Performance Index':
        plt.scatter(df[i],df['Performance Index'])
        plt.title(f'Scatterplot for {i} vs Performance Index')
        plt.show()


# #### from above graphs we see linear relationship between prev scores and Performance Index

# ## Multivariate Analysis
# * Pairplot

# In[55]:


sns.pairplot(data=df,hue='Performance Index')
plt.show()


# In[56]:


# Seperate Performance index as Y feature
X = df.drop(labels=['Performance Index'],axis=1)
Y = df[['Performance Index']]


# In[57]:


X.head()


# In[58]:


Y.head()


# In[71]:


cat1,con1 = catconsep(X)
cat1


# In[72]:


con1


# ## Build Pipeline to train the model

# In[76]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[77]:


num_pipe = Pipeline(steps=[('SimpleImputer',SimpleImputer(strategy='mean')),
                           ('Scaler',StandardScaler())])
cat_pipe = Pipeline(steps = [('SimpleImputer', SimpleImputer(strategy='most_frequent')),
                             ('OHE',OneHotEncoder(handle_unknown='ignore'))])
pre = ColumnTransformer([('num',num_pipe,con1),
                        ('cat',cat_pipe,cat1)])


# In[82]:


X_pre = pre.fit_transform(X)


# In[83]:


X_pre


# In[84]:


cols = pre.get_feature_names_out()


# In[85]:


cols


# In[87]:


X_pre = pd.DataFrame(X_pre,columns=cols)
X_pre.head()


# In[88]:


X_pre.shape


# ## Train-test Split

# In[89]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X_pre,Y,test_size=0.2,random_state=21)


# In[90]:


xtrain.shape


# In[91]:


xtest.shape


# In[93]:


ytrain.shape


# In[94]:


ytrain.head()


# ## Linear Regression Model

# In[95]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain,ytrain)


# In[97]:


model.score(xtrain,ytrain)


# In[98]:


model.score(xtest,ytest)


# In[105]:


#evaluate the model
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
def model_evaluation(xtrain,ytrain,xtest,ytest,model):
    
    ypred_tr = model.predict(xtrain)
    ypred_ts = model.predict(xtest)
    
    tr_mae = mean_absolute_error(ytrain,ypred_tr)
    tr_mse = mean_squared_error(ytrain,ypred_tr)
    tr_rmse = tr_mse**(1/2)
    tr_r2 = r2_score(ytrain,ypred_tr)
    
    ts_mae = mean_absolute_error(ytest,ypred_ts)
    ts_mse = mean_squared_error(ytest,ypred_ts)
    ts_rmse = ts_mse**(1/2)
    ts_r2 = r2_score(ytest,ypred_ts)
    
    print('Training Model Evaluation\n')
    print(f"Mean Absolute Error: {tr_mae:.2f}")
    print(f"Mean Squared Error: {tr_mse:.2f}")
    print(f"Root Mean Squared Error: {tr_rmse:.2f}")
    print(f"R2 Score: {tr_r2:.4f}")
    print ('----------------------------------')
    print('\nTesting Model Evaluation\n')
    print(f"Mean Absolute Error: {ts_mae:.2f}")
    print(f"Mean Squared Error: {ts_mse:.2f}")
    print(f"Root Mean Squared Error: {ts_rmse:.2f}")
    print(f"R2 Score: {ts_r2:.4f}")
    
    
    


# In[106]:


model_evaluation(xtrain,ytrain,xtest,ytest,model)


# In[108]:


preds = model.predict(X_pre)
preds


# In[109]:


df['Predicted Performance Index'] = preds


# In[110]:


df.head()


# In[111]:


ytest.head()


# In[115]:


df1 =df[['Performance Index']]
df1


# In[116]:


df1['Predcited Performance Index'] = preds
df1


# In[ ]:




