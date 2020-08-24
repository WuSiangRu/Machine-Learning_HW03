from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


iris=datasets.load_iris()
X=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
accuracy=[]
for i in range(1,5):
    forest=RandomForestClassifier(criterion="entropy",n_estimators=10,random_state=1,max_features=i)
    forest.fit(X_train,y_train)
    res=forest.predict(X_test)
    accuracy.append(accuracy_score(y_test,res))
accuracy=np.round(accuracy,2)


plt.plot(range(1,5),accuracy)
plt.title("Random Forest",fontsize=18)
plt.xlabel("Number of features",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.xticks([1,2,3,4])
plt.yticks([0.92,0.93,0.95])
plt.grid("on")
plt.show()







