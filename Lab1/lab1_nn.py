import numpy as np
import random
from sklearn.metrics import accuracy_score

def bit_func(x, y):
  return x | y & ~y

def step_func(x):
  if(x > 0):
    return 1
  else:
    return 0

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias
  def get_sum(self, x):
    sum = self.bias
    for i in range(len(self.weights)):
      sum += self.weights[i] * x[i]
    return sum
  def feed(self, x):
    y = step_func(self.get_sum(x))
    return y
  def print_params(self):
    n_str = ''
    for i in range(len(self.weights)):
      n_str += str(self.weights[i]) + " "
    print(n_str + str(self.bias))

class Network:
  def __init__(self):
    self.n1 = Neuron([random.randint(-5, 5), random.randint(-5, 5)], random.randint(-5, 5))
    self.n2 = Neuron([random.randint(-5, 5), random.randint(-5, 5)], random.randint(-5, 5))
    self.o1 = Neuron([random.randint(-5, 5), random.randint(-5, 5)], random.randint(-5, 5))

    # self.n1 = Neuron([-1, 1], 1)
    # self.n2 = Neuron([1, -1], 1)
    # self.o1 = Neuron([1, 1], 1)

    # self.n1 = Neuron([1, 1], 1)
    # self.n2 = Neuron([1, 1], 1)
    # self.o1 = Neuron([1, 1], 1)
  def printNeurons(self):
    print("[" + str(round(self.n1.weights[0], 2)) + " " + str(round(self.n1.weights[1], 2)) + " " + str(round(self.n1.bias, 2)) + " " +
                str(round(self.n2.weights[0], 2)) + " " + str(round(self.n2.weights[1], 2)) + " " + str(round(self.n2.bias, 2)) + " " +
                str(round(self.o1.weights[0], 2)) + " " + str(round(self.o1.weights[1], 2)) + " " + str(round(self.o1.bias, 2)) + "]")
  def predict(self, data):
    y_pred = []
    for x in data:
      h1 = self.n1.feed(x)
      h2 = self.n2.feed(x)
      h = [h1, h2]
      y_pred.append(self.o1.feed(h))
    return(y_pred)
  def trainHebb(self, data, y_trues):
    epochs = 10
    self.printNeurons()
    for epoch in range(epochs):
      for x, y_true in zip(data, y_trues):
        h1 = self.n1.feed(x)
        h2 = self.n2.feed(x)
        h = [h1, h2]
        out1 = self.o1.feed(h)

        if(out1 > y_true):
          for i in range(len(x)):
            self.n1.weights[i] -= x[i]
            self.n2.weights[i] -= x[i]
            self.o1.weights[i] -= x[i]
          self.n1.bias -= 1
          self.n2.bias -= 1
          self.o1.bias -= 1
        elif(out1 < y_true):
          for i in range(len(x)):
            self.n1.weights[i] += x[i]
            self.n2.weights[i] += x[i]
            self.o1.weights[i] += x[i]
          self.n1.bias += 1
          self.n2.bias += 1
          self.o1.bias += 1
          
      y_preds = self.predict(data)
      acc = accuracy_score(y_trues, y_preds)
      print("Epoch %d acc: %.3f" % (epoch, acc))
      self.printNeurons()
    y_preds = self.predict(data)
    acc = accuracy_score(y_trues, y_preds)
    print("acc: " + str(acc))
    self.printNeurons()
      
all_data = np.array([
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1],
])

all_y_trues = np.array([
  0,
  1,
  0,
  1,
])

network = Network()
network.trainHebb(all_data, all_y_trues)