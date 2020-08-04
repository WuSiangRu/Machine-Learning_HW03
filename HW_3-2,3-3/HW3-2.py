from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def versiontuple(v):
    return tuple(map(int,(v.split("."))))

def plot_decision_regions(X,y, classifier, test_idx=None,resolution=0.02):
    markers = ('s','x','o','^','v')
    colors =('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl ,1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    if test_idx:
        if not versiontuple(np.__version__)>=versiontuple('1.9.0'):
            X_test,y_test=X[list(test_idx), :],y[test_idx]
        else:
            X_test,y_test=X[test_idx, :],y[test_idx]

        plt.scatter(X_test[:,0],X_test[:,1],c='',alpha=1.0,linewidths=1,marker='o',s=55,label='test set')



iris=datasets.load_iris()
X=iris.data[:,[2,3]]
y=iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

#Outlier設定:將標準化後的第一筆數據的特徵修改成[1,-1],使其成為離群值。
#print(X_train_std[0])
X_train_std[0]=[1,-1]
#print(X_train_std[0])

from sklearn.svm import SVC
#C值=0.1
svm=SVC(kernel='linear',C=0.1,random_state=0)
svm.fit(X_train_std,y_train)
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plt.subplot(2,2,1)
plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title("C=0.1")

#C值=1.0
svm=SVC(kernel='linear',C=1.0,random_state=0)
svm.fit(X_train_std,y_train)
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plt.subplot(2,2,2)
plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title("C=1")

#C值=10.0
svm=SVC(kernel='linear',C=10.0,random_state=0)
svm.fit(X_train_std,y_train)
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plt.subplot(2,2,3)
plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title("C=10")

#C值=100.0
svm=SVC(kernel='linear',C=100.0,random_state=0)
svm.fit(X_train_std,y_train)
X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plt.subplot(2,2,4)
plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title("C=100")


plt.show()