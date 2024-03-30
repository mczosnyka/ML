#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urllib.request import urlretrieve
import tarfile
import os
import gzip
import shutil
url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz'
filename = 'housing.tgz'
urlretrieve(url, filename)
with tarfile.open(filename, 'r:gz') as tar:
    tar.extractall()


# In[2]:


extracted_filename='housing.csv'
compressed_filename = 'housing.csv.gz'
with open(extracted_filename, 'rb') as f_in, gzip.open(compressed_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('housing.csv.gz')
df.head()
df.info()


# In[4]:


df["ocean_proximity"].info()


# In[5]:


df["ocean_proximity"].value_counts()


# In[6]:


df["ocean_proximity"].describe()


# In[7]:


df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')


# In[8]:


df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(7,4))
plt.savefig('obraz2.png')


# In[9]:


df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(7,3), colorbar=True, s=df["population"]/100, label="population", c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')


# In[10]:


s = df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False)
s.reset_index().rename(columns={"index": "atrybut", "median_house_value": "wspolczynnik_korelacji"}).to_csv('korelacja.csv', index=False) 


# In[11]:


sns.pairplot(df)


# In[12]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(train_set), len(test_set)


# In[13]:


train_set.head()


# In[14]:


test_set.head()


# In[15]:


s_train = train_set.corr(numeric_only=True)
s_train.head(10)


# In[16]:


s_test = test_set.corr(numeric_only=True)
s_test.head(10)


# In[17]:


import pickle


# In[18]:


pickle.dump(train_set, open('train_set.pkl', 'wb'))


# In[19]:


pickle.dump(test_set, open('test_set.pkl', 'wb'))

