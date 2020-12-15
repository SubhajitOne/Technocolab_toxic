#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt


# In[26]:


train_1 = pd.read_csv('../input/toxic-datas/train.csv')
test_1 = pd.read_csv('../input/toxic-datas/test.csv')


# In[27]:


train_1.head()


# In[28]:


train_1.shape


# In[29]:


#EDA
plt.pie(train_1.toxic.value_counts(normalize=True))
plt.show()


# In[30]:


data_count = train_1.iloc[:,3:].sum()


# In[31]:


plt.figure(figsize=(8,5))
plt.bar(data_count.index,data_count.values)
plt.show()


# In[32]:


train = train_1.copy()
test = test_1.copy()


# ## Data Preprocessing

# In[33]:


#Cnverting the comment_text to lower case.
train['comment_text'] = train['comment_text'].apply(lambda x: x.lower())
test['comment_text'] = test['comment_text'].apply(lambda x: x.lower())


# In[34]:


#Remove Punctuation
def remove_punc(text):
  return re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)

train['comment_text'] = train['comment_text'].apply(lambda x: remove_punc(x))
test['comment_text'] = test['comment_text'].apply(lambda x: remove_punc(x))


# In[35]:


#Removing numbers with letters attach to them
train['comment_text'] = train['comment_text'].apply(lambda x: re.sub('\w*\d\w*', ' ', x))
test['comment_text'] = test['comment_text'].apply(lambda x: re.sub('\w*\d\w*', ' ', x))


# In[36]:


#Removing '/n '
train['comment_text'] = train['comment_text'].apply(lambda x: re.sub('\n', ' ', x))
test['comment_text'] = test['comment_text'].apply(lambda x: re.sub('\n', ' ', x))


# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[38]:


tf = TfidfVectorizer(stop_words='english')
X_train = tf.fit_transform(train['comment_text'])
X_test = tf.transform(test['comment_text'])


# ## Model

# In[41]:


from sklearn.linear_model import LogisticRegression


# In[44]:


model=[]
i=0
for a in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    y = train[a]
    model.append(LogisticRegression(solver='sag'))
    model[i].fit(X_train, y)
    i=i+1


# ## Prediction

# In[45]:


i=0
for a in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    print("Column:",a)
    pred_pro = model[i].predict_proba(X_train)[:,1]
    print(pred_pro)
    i=i+1


# In[57]:


i=0
for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    test[col]=model[i].predict_proba(X_test)[:, 1]
    i=i+1
test


# ## Pickle File

# In[59]:


import pickle


# In[61]:


pk=open("model.pkl", "wb")
pickle.dump(model,pk)
pickle.dump(tf,pk)


# In[ ]:




