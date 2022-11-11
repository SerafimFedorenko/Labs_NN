import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from PIL import Image, ImageChops
import time

from network import *

plt.figure(figsize=(10,10))

def get_img(path):
  test_img = Image.open(path)
  test_img = test_img.resize((24, 24))
  plt.subplot(1,5,i)
  plt.imshow(test_img)
  test_x = np.array(test_img, np.float32)
  test_x = test_x / 255.0
  print(test_x.shape)
  return test_x

x_train = []
y_train = []

x_test = []
y_test = []

path = 'D:/Лабы 3 курс/ВвНС/Lab4/Data/Train/Apples/'
for i in range(1,6,1):
  x_train.append(get_img(path + str(i) +'.png'))
  y_train.append(0)

plt.show()

path = 'D:/Лабы 3 курс/ВвНС/Lab4/Data/Test/Apples/'
for i in range(1,4,1):
  x_test.append(get_img(path + str(i) +'.png'))
  y_test.append(0)

plt.show()

path = 'D:/Лабы 3 курс/ВвНС/Lab4/Data/Train/Lemons/'
for i in range(1,6,1):
  x_train.append(get_img(path + str(i) +'.png'))
  y_train.append(1)

plt.show()

path = 'D:/Лабы 3 курс/ВвНС/Lab4/Data/Test/Lemons/'
for i in range(1,4,1):
  x_test.append(get_img(path + str(i) +'.png'))
  y_test.append(1)

plt.show()

path = 'D:/Лабы 3 курс/ВвНС/Lab4/Data/Train/Pears/'
for i in range(1,6,1):
  x_train.append(get_img(path + str(i) +'.png'))
  y_train.append(2)

plt.show()

path = 'D:/Лабы 3 курс/ВвНС/Lab4/Data/Test/Pears/'
for i in range(1,4,1):
  x_test.append(get_img(path + str(i) +'.png'))
  y_test.append(2)

plt.show()

x_tr = []
x_tst = []

for i in range(len(x_train)):
  x = []
  for j in range(len(x_train[0])):
    for k in range(len(x_train[0][0])):
      x.extend(1 - x_train[i][j][k][:3])
  x_tr.append(x)
print(x_tr)

for i in range(len(x_test)):
  x = []
  for j in range(len(x_test[0])):
    for k in range(len(x_test[0][0])):
      x.extend(1 - x_test[i][j][k][:3])
  x_tst.append(x)

print(x_tst)

print(y_to_arrays(y_test))
print(arrays_to_y(y_to_arrays(y_test)))
print(y_to_arrays([0,1,1,0]))
network = NetworkMC(2, 16, 8, 4, 2)
x_xor = [[0,0],[0,1],[1,0],[1,1]]
y_xor = [0,1,1,0]

print("train on xor...")
epochs, errors = network.train_grad_full_batch(x_xor, y_xor, 0.1, 10000)
plt.plot(epochs, errors)
plt.show()

print("train on pictures...")
start_time = time.time()
network = NetworkMC(1728, 256, 128, 64, 3)
epochs, errors = network.train_grad_full_batch(x_tr, y_train, 0.01, 100)
print("time of working: " + str(round(time.time() - start_time, 2)) + " seconds")
plt.plot(epochs, errors)
plt.show()

print("Tests")
y_pred = network.predict(x_tst)
y_pred_classes = []
print(y_pred)
for i in range(len(y_pred)):
  y_pred_classes.append(get_res_multy(y_pred[i]))
acc = err_acc(y_test, y_pred_classes, 0.1)
print("acc:" + str(acc))
ress = ['Яблоко', 'Лимон', 'Груша']
plt.figure(figsize=(15,5))
for i in range(9):
  plt.subplot(1, 10, i + 1)
  plt.title(ress[get_res_multy(y_pred[i])])
  plt.imshow(x_test[i])

plt.show()