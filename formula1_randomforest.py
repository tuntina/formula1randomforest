#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt



#sürücü listesi okutuldu
driver = pd.read_csv("C:\\Users\\evata\\formula1\\drivers.csv")
driver.head()




#sürücü listesi okutuldu
circuits = pd.read_csv("C:\\Users\\evata\\formula1\\circuits.csv")
circuits.head()




#sonuc listesi okutuldu
results = pd.read_csv("C:\\Users\\evata\\formula1\\results.csv")
results.head()




#Yarışlar veri seti okundu
races = pd.read_csv("C:\\Users\\evata\\formula1\\races.csv")
races.head()



#yarış durumları veri seti okutuldu
status = pd.read_csv("C:\\Users\\evata\\formula1\\status.csv")
status.head()



#results verilerinde önemsizler silinir

results = results.drop(columns = ["time", "constructorId","positionOrder","positionText","points","fastestLapSpeed","fastestLapTime", "fastestLap"])
results.head()



#driver verilerinde önemsizler silinir

driver = driver.drop(columns = ["url"])
driver.head()



#races verilerinde önemsizler silinir
races = races.drop(columns = ['url', 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', 'quali_date', 'quali_time', 'sprint_date', 'sprint_time', 'time'])
races.head()



#status ve results veri seti statusId sütunu ile birleştirildi
resultscopy = results.copy()
data = results.merge(status, on = "statusId")




# birleştirilen veri seti racesveri seti ile raceId verisi ile birleştirildi.
data = races.merge(data, on = "raceId")

data.head()




#son on yıllık veriler ile çalışılması için 2012 ve üzeriyıllar dikkate alındı
data=data[data["year"]>=2012]



# yarışı bitiren yarıçılar dikkate alındı
data=data[data["status"] == "Finished"]



#driver veri setini merge komutu ile birleştirildi.
data=driver.merge(data,on = "driverId" )
data



# birleştirilen veri seti racesveri seti ile raceId verisi ile birleştirildi.
data = circuits.merge(data, on = "circuitId")

data.head()



data = data.drop(["circuitRef","location","lat","lng","alt","url", "name_x","number_x","number_y"], axis = 1)




#sütunların isimleri değişti
data.rename(columns ={'name':'pistname'},inplace=True)
data.rename(columns ={'number':'carnumber'},inplace=True)
data.rename(columns ={'name_y':'pistname'},inplace=True)

data.head()


# veriler içinde boş değer olmadığı ve veri sayılarında benzerlik olduğu görüldü
data.info()



#"\N" yazan satırlar çıkarıldı
data.drop(data[data.values == "\\N"].index, inplace=True)



#object tipindeki verilerin string,int,float a dnüştürülmesi
data['position'] = data['position'].astype(np.int64)
data.astype({'code':'string', 'forename':'string', 'surname':'string','dob':'string', 'nationality':'string'}).dtypes
data['date'] =  pd.to_datetime(data['date'], infer_datetime_format=True)



data.columns



#random forest için kullanılmayacak veriler silinip etiket verileri ve değişkenler ayrı değişkenlerde tanımlandı.
X = data.drop(["rank", "status", "forename" ,"surname","date","dob","nationality","round","driverRef",
            "resultId", "statusId","laps","code","milliseconds","position","pistname",
              "country","raceId"], axis=1)
y = data["position"]




#random forest modeli
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=44, max_features="auto", random_state=44)
rf_model.fit(X_train, y_train)




predictions = rf_model.predict(X_test)
predictions




y_test




drivers_list = data[["driverId","code","forename","surname"]]
drivers_list




#tekrarlı olanlar cıkartılır

drivers_list.drop_duplicates(subset=["driverId", "code", "forename","surname"]) 




race_list =data[["raceId","pistname"]]
race_list




race_list.drop_duplicates(subset=["raceId", "pistname"]) 




pist_list =data[["circuitId","pistname","country"]]




#tekrarlı olanlar cıkartılır
pist_list.drop_duplicates(subset=["circuitId","pistname","country"]) 





#circuitId,driverId,year,grid  girilerek sürücünün belirtilen pistte belirtilen yılda ve belirtilen poziyonda
#başlarsa kaçıncı olur tahmini yapılır.
#Turkish Grand Prix de Hamilton 2020 yılında 6. olarak başlarsa makine öğrenmesi ile 1. olabileceğini tahmin etti.

predict= [[5,1,2020,6]]


rf_model.predict(prediction)


#German Grand Prix de Verstapen 2022 yılında 9. olarak başlarsa makine öğrenmesi ile 1. olabileceğini tahmin etti.

predict= [[10,830,2022,9]]
rf_model.predict(predict)




