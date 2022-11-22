#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[42]:


#sürücü listesi okutuldu
driver = pd.read_csv("C:\\Users\\evata\\formula1\\drivers.csv")
driver.head()


# In[7]:


#sürücü listesi okutuldu
circuits = pd.read_csv("C:\\Users\\evata\\formula1\\circuits.csv")
circuits.head()


# In[8]:


#sonuc listesi okutuldu
results = pd.read_csv("C:\\Users\\evata\\formula1\\results.csv")
results.head()


# In[9]:


#Yarışlar veri seti okundu
races = pd.read_csv("C:\\Users\\evata\\formula1\\races.csv")
races.head()


# In[10]:


#yarış durumları veri seti okutuldu
status = pd.read_csv("C:\\Users\\evata\\formula1\\status.csv")
status.head()


# In[11]:


#results verilerinde önemsizler silinir

results = results.drop(columns = ["time", "constructorId","positionOrder","positionText","points","fastestLapSpeed","fastestLapTime", "fastestLap"])
results.head()


# In[12]:


#driver verilerinde önemsizler silinir

driver = driver.drop(columns = ["url"])
driver.head()


# In[13]:


#races verilerinde önemsizler silinir
races = races.drop(columns = ['url', 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', 'quali_date', 'quali_time', 'sprint_date', 'sprint_time', 'time'])
races.head()


# In[14]:


#status ve results veri seti statusId sütunu ile birleştirildi
resultscopy = results.copy()
data = results.merge(status, on = "statusId")


# In[15]:


# birleştirilen veri seti racesveri seti ile raceId verisi ile birleştirildi.
data = races.merge(data, on = "raceId")

data.head()


# In[16]:


#son on yıllık veriler ile çalışılması için 2012 ve üzeriyıllar dikkate alındı
data=data[data["year"]>=2012]


# In[17]:


# yarışı bitiren yarıçılar dikkate alındı
data=data[data["status"] == "Finished"]


# In[18]:


#driver veri setini merge komutu ile birleştirildi.
data=driver.merge(data,on = "driverId" )
data


# In[19]:


# birleştirilen veri seti racesveri seti ile raceId verisi ile birleştirildi.
data = circuits.merge(data, on = "circuitId")

data.head()


# In[20]:


data = data.drop(["circuitRef","location","lat","lng","alt","url", "name_x","number_x","number_y"], axis = 1)


# In[21]:


#sütunların isimleri değişti
data.rename(columns ={'name':'pistname'},inplace=True)
data.rename(columns ={'number':'carnumber'},inplace=True)
data.rename(columns ={'name_y':'pistname'},inplace=True)

data.head()


# In[22]:


data.info()


# In[23]:


#"\N" yazan satırlar çıkarıldı
data.drop(data[data.values == "\\N"].index, inplace=True)


# In[24]:


#object tipindeki verilerin string,int,float a dnüştürülmesi
data['position'] = data['position'].astype(np.int64)
data.astype({'code':'string', 'forename':'string', 'surname':'string','dob':'string', 'nationality':'string'}).dtypes
data['date'] =  pd.to_datetime(data['date'], infer_datetime_format=True)


# In[25]:


data.columns


# In[26]:


#random forest için kullanılmayacak veriler silinip etiket verileri ve değişkenler ayrı değişkenlerde tanımlandı.
X = data.drop(["rank", "status", "forename" ,"surname","date","dob","nationality","round","driverRef",
            "resultId", "statusId","laps","code","milliseconds","position","pistname",
              "country","raceId"], axis=1)
y = data["position"]


# In[27]:


X


# In[28]:


#random forest modeli
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=44, max_features="auto", random_state=44)
rf_model.fit(X_train, y_train)


# In[29]:


predictions = rf_model.predict(X_test)
predictions


# In[30]:


y_test


# In[31]:


drivers_list = data[["driverId","code","forename","surname"]]
drivers_list


# In[32]:


#tekrarlı olanlar cıkartılır

drivers_list.drop_duplicates(subset=["driverId", "code", "forename","surname"]) 


# In[33]:


race_list =data[["raceId","pistname"]]
race_list


# In[34]:


race_list.drop_duplicates(subset=["raceId", "pistname"]) 


# In[35]:


pist_list =data[["circuitId","pistname","country"]]


# In[36]:


#tekrarlı olanlar cıkartılır
pist_list.drop_duplicates(subset=["circuitId","pistname","country"]) 


# In[37]:


X


# In[45]:


#circuitId,driverId,year,grid  girilerek sürücünün belirtilen pistte belirtilen yılda ve belirtilen poziyonda
#başlarsa kaçıncı olur tahmini yapılır.

prediction= [[5,4,2018,7]]


# In[46]:


rf_model.predict(prediction)


# In[47]:


rfc = RandomForestClassifier(random_state=0)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score

print('Model accuracy score : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[ ]:




