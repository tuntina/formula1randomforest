#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:


#sürücü listesi okutuldu
driver = pd.read_csv("C:\\Users\\evata\\formula1\\drivers.csv")
driver.head()


# In[3]:


#sürücü listesi okutuldu
circuits = pd.read_csv("C:\\Users\\evata\\formula1\\circuits.csv")
circuits.head()


# In[4]:


#sonuc listesi okutuldu
results = pd.read_csv("C:\\Users\\evata\\formula1\\results.csv")
results.head()


# In[5]:


#Yarışlar veri seti okundu
races = pd.read_csv("C:\\Users\\evata\\formula1\\races.csv")
races.head()


# In[6]:


#yarış durumları veri seti okutuldu
status = pd.read_csv("C:\\Users\\evata\\formula1\\status.csv")
status.head()


# In[7]:


#results verilerinde önemsizler silinir

results = results.drop(columns = ["time", "constructorId","positionOrder","positionText","points","fastestLapSpeed","fastestLapTime", "fastestLap"])
results.head()


# In[8]:


#driver verilerinde önemsizler silinir

driver = driver.drop(columns = ["url"])
driver.head()


# In[9]:


#races verilerinde önemsizler silinir
races = races.drop(columns = ['url', 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', 'quali_date', 'quali_time', 'sprint_date', 'sprint_time', 'time'])
races.head()


# In[10]:


#status ve results veri seti statusId sütunu ile birleştirildi
resultscopy = results.copy()
data = results.merge(status, on = "statusId")


# In[11]:


# birleştirilen veri seti racesveri seti ile raceId verisi ile birleştirildi.
data = races.merge(data, on = "raceId")

data.head()


# In[12]:


#son on yıllık veriler ile çalışılması için 2012 ve üzeriyıllar dikkate alındı
data=data[data["year"]>=2012]


# In[13]:


# yarışı bitiren yarıçılar dikkate alındı
data=data[data["status"] == "Finished"]


# In[14]:


#driver veri setini merge komutu ile birleştirildi.
data=driver.merge(data,on = "driverId" )
data


# In[15]:


# birleştirilen veri seti racesveri seti ile raceId verisi ile birleştirildi.
data = circuits.merge(data, on = "circuitId")

data.head()


# In[16]:


data = data.drop(["circuitRef","location","lat","lng","alt","url", "name_x","number_x","number_y"], axis = 1)


# In[17]:


#sütunların isimleri değişti
data.rename(columns ={'name':'pistname'},inplace=True)
data.rename(columns ={'number':'carnumber'},inplace=True)
data.rename(columns ={'name_y':'pistname'},inplace=True)

data.head()


# In[18]:


data.info()


# In[19]:


#"\N" yazan satırlar çıkarıldı
data.drop(data[data.values == "\\N"].index, inplace=True)


# In[20]:


#object tipindeki verilerin string,int,float a dnüştürülmesi
data['position'] = data['position'].astype(np.int64)
data.astype({'code':'string', 'forename':'string', 'surname':'string','dob':'string', 'nationality':'string'}).dtypes
data['date'] =  pd.to_datetime(data['date'], infer_datetime_format=True)



# In[21]:


data.columns


# In[22]:


#random forest için kullanılmayacak veriler silinip etiket verileri ve değişkenler ayrı değişkenlerde tanımlandı.
X = data.drop(["rank", "status", "forename" ,"surname","date","dob","nationality","round","driverRef",
            "resultId", "statusId","laps","code","milliseconds","position","pistname",
              "country","raceId"], axis=1)
y = data["position"]


# In[23]:


X


# In[24]:


#random forest modeli
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=44, max_features="auto", random_state=44)
rf_model.fit(X_train, y_train)


# In[25]:


predictions = rf_model.predict(X_test)
predictions


# In[26]:


y_test


# In[27]:


drivers_list = data[["driverId","code","forename","surname"]]
drivers_list


# In[28]:


#tekrarlı olanlar cıkartılır

drivers_list.drop_duplicates(subset=["driverId", "code", "forename","surname"]) 


# In[29]:


race_list =data[["raceId","pistname"]]
race_list


# In[30]:


race_list.drop_duplicates(subset=["raceId", "pistname"]) 


# In[31]:


pist_list =data[["circuitId","pistname","country"]]


# In[32]:


#tekrarlı olanlar cıkartılır
pist_list.drop_duplicates(subset=["circuitId","pistname","country"]) 


# In[33]:


X


# In[44]:


#circuitId,driverId,year,grid  girilerek sürücünün belirtilen pistte belirtilen yılda ve belirtilen poziyonda
#başlarsa kaçıncı olur tahmini yapılır.
#Turkish Grand Prix de Hamilton 2020 yılında 6. olarak başlarsa makine öğrenmesi ile 1. olabileceğini tahmin etti.
predict= [[5,1,2020,6]]


# In[42]:


rf_model.predict(predict)


# In[36]:


#German Grand Prix de Verstapen 2022 yılında 9. olarak başlarsa makine öğrenmesi ile 1. olabileceğini tahmin etti.

predict= [[10,830,2022,9]]


# In[37]:


rf_model.predict(predict)


# In[43]:


rfc = RandomForestClassifier(random_state=0)

rfc.fit(X_train, y_train)

# test sonuclarından tahmin etme

y_pred = rfc.predict(X_test)

# doğruluk analizi

from sklearn.metrics import accuracy_score

print('Model accuracy score : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#fakat formula1 yarışlarında yarışa etki eden birçok faktör bulunuyor (araç,lastik vs.). Bu yüzden bu modelde yüzde 20 doğruluk 
#vermiş oldu.


# In[ ]:




