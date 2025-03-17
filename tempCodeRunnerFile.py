# sklearn : ML Library
from sklearn.datasets import load_breast_cancer
import pandas as pd

# (2)
from sklearn.neighbors import KNeighborsClassifier
#(3)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# (4)
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# (1) Veri Seti İncelemesi
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df["target"] = cancer.target

# (2) Makine Öğrenmesi Modelinin Seçilmesi - KNN 
# (3) Modelin Train Edilmesi
x = cancer.data  #Features
y = cancer.target  #target

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Ölçeklendirme
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# KNN modeli oluştur ve Train et
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)  #fit func verimizi(samples + target) kullanarak knn algortimasını eğitir

# (4) Sonuçların Değerlendirilmesi : test
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(conf_matrix)

# (5) Hiperparametre Ayarlaması
"""
KNN : Hyperparameter = K 
K : 1,2,3, ... N
Accuracy : %A, %B, %C.....

"""
accuracy_values = []
k_values = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)
    
plt.figure()
plt.plot(k_values, accuracy_values, marker = "o", linestyle = "-")
plt.title("K değerine göre doğruluk")
plt.xlabel("K değeri")
plt.ylabel("Doğruluk")
plt.xticks(k_values)  #k değerlerini değiştirmek
plt.grid(True)    #arka plana kare ekler

# %%
# yeni bir section oluşturduk

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


x = np.sort(5 * np.random.rand(40, 1), axis = 0)  #features
y = np.sin(x).ravel()   #target
#plt.plot(x, y)    #boşluk yoktu
#plt.scatter(x,y)    #boşluklu yapı için

#add noise
"""
8 tane rastgele sayıyı çarp 1 den çıkar x te 5 erli yapı oluştur
"""
y[::5] += 1 * (0.5 - np.random.rand(8))

#plt.scatter(x,y)
T  = np.linspace(0,5, 500)[:, np.newaxis]

for i, weight in  enumerate(["uniform", "distance"]) :
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(x,y).predict(T)
    
    plt.subplot(2, 1,i + 1 )
    plt.scatter(x, y, color = "green", label = "data")
    plt.plot(T, y_pred, color = "blue", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN REGRESSOR weights = {}".format(weight))
    
plt.tight_layout()
plt.show()
