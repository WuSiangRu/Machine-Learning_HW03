from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris=datasets.load_iris()
X=iris.data[:,[0,3]]
y=iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#將樣本特徵進行標準化
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
acc_1=[]
acc_2=[]
for i in range(1,10):
    knn=KNeighborsClassifier(n_neighbors=i,p=2,metric="minkowski")
    knn.fit(X_train_std,y_train)
    res_1=knn.predict(X_test_std)
    acc_1.append(accuracy_score(y_test,res_1))
acc_1=np.round(acc_1,2)

#執行結果圖2[10,15,20~40]
for j in range(10,45,5):
    knn = KNeighborsClassifier(n_neighbors=j, p=2, metric="minkowski")
    knn.fit(X_train_std, y_train)
    res_2 = knn.predict(X_test_std)
    acc_2.append(accuracy_score(y_test, res_2))
acc_2 = np.round(acc_2, 2)

#結果圖1
plt.plot(range(1,10),acc_1)
plt.title("KNeighborsClassifier")
plt.xlabel("Number of nighbors")
plt.ylabel("Accuracy")
plt.yticks([0.93,0.96,0.98])
plt.grid("on")
plt.show()

#結果圖2
plt.plot(range(10,45,5),acc_2)
plt.title("KNeighborsClassifier")
plt.xlabel("Number of nighbors")
plt.ylabel("Accuracy")
plt.yticks([0.91,0.93,0.96,0.98])
plt.grid("on")
plt.show()