#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


import nltk
import pandas as pd
import numpy as np


# In[3]:


train = pd.read_csv("train.txt", delimiter=';', header=None, names=['sentence','label'])
test = pd.read_csv("test.txt", delimiter=';', header=None, names=['sentence','label'])
val = pd.read_csv("val.txt", delimiter=';', header=None, names=['sentence','label'])


# In[4]:


df_data = pd.concat([train, test,val])
df_data


# In[5]:


df_data.to_csv (r'exportdata.txt', index=False)
dt_data =  pd.read_csv("exportdata.txt")
dt_data


# In[6]:


# pd.value_counts(dt_data['label']).plot.bar()


# In[7]:


dt_data['label'].value_counts()


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english', ngram_range=(1,1), tokenizer = token.tokenize)
text = cv.fit_transform(dt_data['sentence'])


# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(text,dt_data['label'], test_size=0.30, random_state=5)


# In[10]:


from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)


# In[11]:


predicted = mnb.predict(X_test)


# In[12]:


from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report


# In[13]:


acc_score = metrics.accuracy_score(predicted,y_test)
prec_score = precision_score(y_test,predicted, average='macro')
recall = recall_score(y_test, predicted,average='macro')
f1 = f1_score(y_test,predicted,average='macro')
matrix = confusion_matrix(y_test,predicted)


# In[14]:


print(str('Accuracy: '+'{:04.2f}'.format(acc_score*100))+'%')
print(str('Precision: '+'{:04.2f}'.format(prec_score*100))+'%')
print(str('Recall: '+'{:04.2f}'.format(recall*100))+'%')
print('F1 Score: ',f1)
print(matrix)


# In[15]:


test_data = ['i feel sick','i am ecstatic my model works', 'i feel shitty', 'i feel lost', 'im petrified', 'i am worried']

test_result = mnb.predict(cv.transform(test_data))

print(test_result)


# In[16]:


# predict=['im not satisfied with the support']    
# test_result = mnb.predict(cv.transform(predict))
# print(test_result)


# In[18]:


import joblib
joblib.dump(mnb, 'NB_spam_model.pkl')


# In[ ]:




