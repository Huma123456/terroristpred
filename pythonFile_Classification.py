#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('globaldataset.csv',encoding = "ISO-8859-1")


# In[3]:


df


# In[4]:


df.columns


# In[5]:


year = df.year.value_counts(ascending=True)


# In[6]:


y=year.tolist()


# In[7]:


x=df['year'].unique()


# In[8]:


fig, ax = plt.subplots(1, 1)
ax.bar(x,y, align="center", width=0.5, alpha=0.5)
ax.set_title("Num of attacks per year")
ax.set_xlabel('Years')
ax.set_ylabel('Num of Attacks')


# ###### Converting the textual data into numeric data using sklearn label encoder

# In[9]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encode_result = df.apply(label_encoder.fit_transform)


# In[10]:


encode_result


# In[11]:


label_encoder.classes_


# ###### It converts categorical text data into model-understandable numerical data, we use the Label Encoder class. For label encoding, import the LabelEncoder class from the sklearn library, then fit and transform your data.

# In[12]:


X = np.asarray(encode_result[['year','country','region','city','location of attack','targtype','gang_name','motive','weaptype_txt','propvalue','propcomment','dbsource']])
y= np.asarray(encode_result['attacktype'])


# In[13]:


#splitting the data based on 70-30 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ##### Classification Model

# In[14]:


from sklearn.tree import DecisionTreeClassifier
dclassifier = DecisionTreeClassifier()
dclassifier.fit(X_train, y_train)
dclassifier_pred = dclassifier.predict(X_test)
print(accuracy_score(y_test,  dclassifier_pred))


# In[15]:


a=confusion_matrix(y_test,dclassifier_pred)
a1=a.flatten()
x=a1[0:4]
print(x)
#[TP, FN, FP, TN]


# In[16]:


labels = 'TruePositive', 'FalseNegative', 'FalsePositive', 'TrueNegative'
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

# Plot
plt.pie(x ,labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# In[32]:


from sklearn.ensemble import RandomForestClassifier
rclassifier = RandomForestClassifier(n_estimators=1000, random_state=0)
rclassifier.fit(X_train, y_train) 
rclassifier_pred = rclassifier.predict(X_test)
print(accuracy_score(y_test,  rclassifier_pred))


# In[18]:


a=confusion_matrix(y_test,rclassifier_pred)
a1=a.flatten()
x=a1[0:4]
print(x)
#[TP, FN, FP, TN]


# In[19]:


labels = 'TruePositive', 'FalseNegative', 'FalsePositive', 'TrueNegative'
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

# Plot
plt.pie(x, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# In[17]:


predicted=label_encoder.inverse_transform(dclassifier_pred)


# ##### actual vs predicted values of attacktype

# In[18]:


predicted_df = pd.DataFrame(predicted, columns = ['prediction']) 
predicted_df


# In[19]:


data = [df["attacktype"], predicted_df["prediction"]]


# In[20]:


headers = ["Actual Attack", "Predicted Attack"]


# In[21]:


predicted_df = pd.concat(data, axis=1, keys=headers)


# In[25]:


predicted_df


# #### predict gangs which would be active in future : Y=attack type, X=year

# In[23]:


test=['2020','Dominican Republic','Central America & Caribbean','Santo Domingo','Unknown','Named Civilian','MANO-D',
      'Unknown','Unknown','Unknown','Unknown','PGIS']


# In[26]:


encoded=list()
c=0
while c<12:#c = column
    r=0
    val=""
    for i in df.iloc[:,c]:#i = single value of each row and column
        i=str(i)
        #print('above',r,c,i,test[c])
        if i==test[c]:
            val=encode_result.iloc[r,c]
        r=r+1
    if val=="":
        val=44
    encoded.append(val)
    c=c+1
encoded


# In[27]:


pred = dclassifier.predict([encoded])


# In[28]:


pred


# In[29]:


result = label_encoder.inverse_transform(pred)


# In[30]:


print("Attack could be:: ", np.array_str(result))

