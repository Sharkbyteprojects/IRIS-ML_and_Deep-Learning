#!/usr/bin/env python
# coding: utf-8
# You need:
# pip install numpy
# pip install keras
# pip install pandas
# pip install scikit-learn
#
# # Example for programming an `= or !=` with Deep Learning
# 
# ## Import data

# In[1]:


import pandas as pd
df=pd.read_csv("words.csv",
              header=None,
              names=["one","two"])
print(df.count())
dfs=pd.read_csv("sln.csv",
              header=None,
              names=["class"])
print(dfs.count())


# ## Init Deep Learning

# In[2]:


from keras.layers import Input
import numpy as np
from keras.layers import Dense
inputs=Input(shape=(2,))
fc=Dense(10)(inputs)
from keras.models import Model
model=Model(input=inputs,output=fc)
predictionss=Dense(8,activation="softmax")(fc)
predictions=Dense(2,activation="softmax")(predictionss)
model=Model(input=inputs,output=predictions)
print(model.summary())
model.compile(optimizer="adam",
             loss="categorical_crossentropy",
             metrics=["accuracy"])


# ## Get Dataset

# In[3]:


X=np.array(df.values)
y=np.array(dfs['class'].values)
X.shape, y.shape
from keras.utils.np_utils import to_categorical
y=to_categorical(y,2)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)


# ## Train

# In[4]:


model.fit(X_train, y_train,epochs=500,validation_split=0.3)


# ## TESTS

# In[5]:


train_loss, train_accuracy=model.evaluate(X_train, y_train)
print(round(train_loss*100,1), round(train_accuracy*100,1))
test_loss, test_accuracy=model.evaluate(X_test, y_test)
print(round(test_loss*100,1), round(test_accuracy*100,1))


# ## Make Exec Funtion

# In[6]:


def deep(one, twos):
    print("Deep Learning:")
    print("Loaded Numbers: " + str(one)+" and "+str(twos))
    deplearn=model.predict(np.array([[one, twos]]))
    print("Probability for != in percent")
    print(round(deplearn[0][0]*100,1))
    print("Probability for = in percent")
    print(round(deplearn[0][1]*100,1))
def deepsv(one, twos):
    deplearn=model.predict(np.array([[one, twos]]))
    if(deplearn[0][0]>deplearn[0][1]):
        print("Values are !=")
        return False
    else:
        print("Values are ==")
        return True


# ### Run it

# In[7]:


first=112
second=110
deep(first,second)
deepsv(first,second)


# ## Share this:

# - [Run on Binder](https://mybinder.org/v2/gh/Sharkbyteprojects/IRIS-ML_and_Deep-Learning/master?filepath=csv%20based%2F%3D%20or%20not.ipynb)
# - [Code on GitHub](https://github.com/Sharkbyteprojects/IRIS-ML_and_Deep-Learning/tree/master/csv%20based)

# ## Need
# - pandas
# - Keras
# - SKLEARN
# - numpy
# 
