#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import pickle
mnist = fetch_openml('mnist_784', version=1, as_frame = True)


# In[2]:


print((np.array(mnist.data.loc[42]).reshape(28,28) > 0 ).astype(int))


# In[3]:


X = mnist.data
y = mnist.target
X.head()


# In[4]:


y = y.sort_values(ascending = True)
y.head(100)


# In[5]:


X = X.reindex(index = y.index)
X.head(100)


# In[6]:


X_train , X_test = X[:56000] , X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[7]:


print(y_train.unique())
print(y_test.unique())


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[9]:


print(y_train.unique())
print(y_test.unique())


# In[10]:


y_train_0 = (y_train == '0')
y_test_0 = (y_test == '0')
print(y_train_0)


# In[11]:


sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_0)


# In[12]:


print(y_test[y_test == '0'])


# In[13]:


print(y_test)


# In[14]:


print(sgd_clf.predict([X_test.loc[37003],X_test.loc[1],X_test.loc[47046]]))


# In[15]:


y_train_pred = sgd_clf.predict(X_train)
y_test_pred = sgd_clf.predict(X_test)


# In[16]:


acc_train = sum(y_train_pred == y_train_0)/len(y_train_0)
acc_test = sum(y_test_pred == y_test_0)/len(y_test_0)
print(acc_test)
print(acc_train)


# In[17]:


acc = []
acc.append(acc_train)
acc.append(acc_test)
print(acc)


# In[18]:


with open("sgd_acc.pkl", 'wb') as file:
    pickle.dump(acc, file)


# In[19]:


cross_validation_score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring = 'accuracy', n_jobs = -1)
print(cross_validation_score)


# In[20]:


with open("sgd_cva.pkl", 'wb') as file:
    pickle.dump(cross_validation_score, file)


# In[21]:


sgd_clf_all = SGDClassifier(random_state=42)
sgd_clf_all.fit(X_train, y_train)


# In[22]:


y_pred_all_test = sgd_clf_all.predict(X_test)
y_pred_all_train = sgd_clf_all.predict(X_train)


# In[23]:


print(y_pred_all_train)
print(y_pred_all_test)


# In[24]:


acc_all_train = sum(y_pred_all_train == y_train)/len(y_train)
acc_all_test = sum(y_pred_all_test == y_test)/len(y_test)


# In[25]:


print(acc_all_train)
print(acc_all_test)


# In[26]:


conf_test = confusion_matrix(y_test, y_pred_all_test)
print(conf_test)


# In[27]:


with open("sgd_cmx.pkl", 'wb') as file:
    pickle.dump(conf_test, file)


# In[ ]:




