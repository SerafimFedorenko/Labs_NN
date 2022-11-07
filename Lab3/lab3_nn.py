import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC, NuSVC
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from google.colab import drive
drive.mount('/content/drive')

plt.figure(figsize=(10,10))

def get_img(path):
  test_img = Image.open(path)
  test_img = test_img.resize((100, 100))
  plt.subplot(5,5,i)
  plt.imshow(test_img, cmap = "gray")
  test_img = test_img.convert('1')
  test_x = np.array(test_img, np.float32)
  test_x = test_x.reshape([-1, 10000])
  return test_x

x_train = []
y_train = []
  
path = '/content/drive/My Drive/Google Colab/Грузовые/'
for i in range(1,11,1):
  x_train.append(get_img(path + str(i) +'.png')[0])
  y_train.append(1)

path = '/content/drive/My Drive/Google Colab/Легковые/'
for i in range(1,11,1):
  x_train.append(get_img(path + str(i) +'.png')[0])
  y_train.append(0)

print(x_train[0])

x_test = []
y_test = []
path = '/content/drive/My Drive/Google Colab/Тест_Авто/Легковые/'
for i in range(1,3,1):
  x_test.append(get_img(path + str(i) +'.png')[0])
  y_test.append(0)

path = '/content/drive/My Drive/Google Colab/Тест_Авто/Грузовые/'
for i in range(1,3,1):
  x_test.append(get_img(path + str(i) +'.png')[0])
  y_test.append(1)

cVals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

accs = []
acc_trs = []
coef = []

for c in cVals:
  svm = LinearSVC(dual=False, C=c)
  svm.fit(x_train, y_train)
  coef = svm.coef_
  y_pred_tr = svm.predict(x_train)
  y_pred = svm.predict(x_test)
  acc = accuracy_score(y_true=y_test, y_pred=y_pred)
  acc_tr = accuracy_score(y_true=y_train, y_pred=y_pred_tr)

  accs.append(acc)
  acc_trs.append(acc_tr)

for i in range(len(accs)):
  print("Accuracy train: " + str(acc_trs[i]) + ", accuracy test: " + str(accs[i]) + ", c = : " + str(cVals[i]))

plt.subplots(figsize = (10, 5))
plt.semilogx(cVals, acc_trs, '-gD', color = 'red', label = "Train accuracy")
plt.semilogy(cVals, accs, '-gD', color = 'green', label = "Test accuracy")
plt.grid()
plt.legend()
plt.show()

svm = LinearSVC(dual=False, C=0.0001)
svm.fit(x_train, y_train)
coef = svm.coef_
y_pred = svm.predict(x_test)
acc = accuracy_score(y_true=y_test, y_pred=y_pred)

y_pred = svm.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("acc:" + str(acc))
ress = ['Легковой', 'Грузовой']
plt.figure(figsize=(15,5))
for i in range(4):
  plt.subplot(1, 4, i + 1)
  plt.title(ress[y_pred[i]])
  img = np.reshape((x_test[i]), [100, 100])
  plt.imshow(img, cmap="gray")

from math import gamma
param_grid = {'C':[0.01, 0.1, 1, 10, 100], 'gamma':[0.0001, 0.001, 0.01, 0.1],'kernel':['linear', 'rbf', 'poly']}
grid_svm = GridSearchCV(SVC(), param_grid = param_grid)
grid_svm.fit(x_train, y_train)
print(grid_svm.best_params_)

svc = SVC(C = 0.01, gamma = 0.0001, kernel = 'linear')
svc.fit(x_train, y_train)
score = svc.score(x_test, y_test)
print("acc: " + str(score))

from math import gamma
param_grid = {'tol':[0.0001, 0.001, 0.01, 0.1],'kernel':['linear', 'rbf', 'poly']}
grid_svm = GridSearchCV(NuSVC(), param_grid = param_grid)
grid_svm.fit(x_train, y_train)
print(grid_svm.best_params_)

svc = NuSVC(kernel = 'linear')
svc.fit(x_train, y_train)
score = svc.score(x_test, y_test)
print(score)