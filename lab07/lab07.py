#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.target = mnist.target.astype(np.uint8)
X = mnist['data']
y = mnist['target']


# In[2]:


print(X)


# In[3]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sil_scores = []
lista = []

for i in range(8,13):
    kmeans = KMeans(n_clusters = i, random_state = 42, n_init = 10)
    y_pred = kmeans.fit_predict(X)
    lista.append([kmeans, y_pred])
    sil_scores.append(silhouette_score(X, y_pred))

print(sil_scores)


# In[4]:


import pickle
with open('kmeans_sil.pkl', 'wb') as f:
    pickle.dump(sil_scores, f)


# In[5]:


from sklearn.metrics import confusion_matrix


# In[6]:


predictions = []
for k in lista:
    predictions.append(k[1])

cm = confusion_matrix(y, predictions[2])
sett = set()
for row in cm:
    sett.add(np.argmax(row))

print(sett)
with open('kmeans_argmax.pkl', 'wb') as f:
    pickle.dump(list(sett), f)


# In[7]:


lengths = []
for p in range(300):
    for point in X: 
        lng = np.linalg.norm(X[p] - point)
        if lng!=0:
            lengths.append(lng)

lengths.sort()

    


# In[8]:


pickler = lengths[:10]
print(pickler)


# In[9]:


with open('dist.pkl', 'wb') as f:
    pickle.dump(pickler, f)


# In[14]:


s = (pickler[0]+pickler[1]+pickler[2])/3
print(s)


# In[28]:


from sklearn.cluster import DBSCAN
unique_labels = []
for ep in np.arange(s, s+0.1*s, 0.04*s):
    dbscan = DBSCAN(eps=ep)
    dbscan.fit(X)
    print(dbscan)
    print(dbscan.labels_)
    print(ep)
    unique_labels.append(np.unique(dbscan.labels_[dbscan.labels_ != -1]))


# In[31]:


unique_numbers = []
unique_numbers.append(len(unique_labels[0])+1)
unique_numbers.append(len(unique_labels[1])+1)
unique_numbers.append(len(unique_labels[2])+1)
print(unique_numbers)
with open('dbscan_len.pkl', 'wb') as f:
    pickle.dump(unique_numbers, f)


# In[ ]:




