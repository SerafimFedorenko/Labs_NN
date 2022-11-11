import numpy as np
import random
import copy

def y_to_arrays(y_t):
  y_arrays = []
  for y in y_t:
    y_arr = np.zeros(np.array(y_t).max() + 1, dtype = int)
    y_arr = y_arr.tolist()
    y_arr[y] = 1
    y_arrays.append(y_arr)
  return y_arrays

def arrays_to_y(y_arrays):
  y_t = []
  for y in y_arrays:
    y_t.append(np.argmax(y))
  return y_t

def fun_sigmoid(s):
  return 1 / (1 + np.exp(-s))

def fun_der_sigm(s):
  return fun_sigmoid(s) * (1 - fun_sigmoid(s))

def err_acc(pred_y, tst_y, eps):
  hit = 0
  for i in range(len(tst_y)):
    if(np.abs(pred_y[i] - tst_y[i]) < eps): hit = hit + 1
  return hit / len(tst_y)

class Neuron:
  def __init__(self, weightsCount, number = None):
    if number is None:
      self.weights = []
      for i in range(weightsCount):
        self.weights.append(np.random.normal())
    else:
      self.weights = []
      for i in range(weightsCount):
        self.weights.append(number)

  def get_sum(self, x):
    sum = 0
    for i in range(len(self.weights)):
      sum += self.weights[i] * x[i]
    return sum

  def feed(self, x):
    y = fun_sigmoid(self.get_sum(x))
    return y

  def print_params(self):
    n_str = ''
    for i in range(len(self.weights)):
      n_str += str(self.weights[i]) + " "
    print(n_str + str(self.bias))

def err_los_mc(tst_y, pred_y):
  loss = 0
  for i in range(len(tst_y)):
    for j in range(len(tst_y[0])):
      loss += np.square(pred_y[i][j] - tst_y[i][j])
  loss = np.sqrt(loss) / len(tst_y)
  return loss

def add_bias_to_data(data):
  data_with_bias = copy.deepcopy(data)
  for elem in data_with_bias:
    elem.append(1)
  return data_with_bias

class NetworkMC:
    def __init__(self, x_count, l1_count, l2_count, l3_count, y_count):
        self.NeuronsL1 = []
        self.NeuronsL2 = []
        self.NeuronsL3 = []
        self.out_layer = []
        for i in range(l1_count):
            n = Neuron(x_count + 1)
            self.NeuronsL1.append(n)
        for i in range(l2_count):
            n = Neuron(l1_count + 1)
            self.NeuronsL2.append(n)
        for i in range(l3_count):
            n = Neuron(l2_count + 1)
            self.NeuronsL3.append(n)
        for i in range(y_count):
            n = Neuron(l3_count + 1)
            self.out_layer.append(n)

    def predict(self, data):
        data_with_bias = add_bias_to_data(data)
        y_pred = []
        for x in data_with_bias:
            y = []
            res1 = []
            for n in self.NeuronsL1:
                res1.append(n.feed(x))
            res1.append(1)
            res2 = []
            for n in self.NeuronsL2:
                res2.append(n.feed(res1))
            res2.append(1)
            res3 = []
            for n in self.NeuronsL3:
                res3.append(n.feed(res2))
            res3.append(1)
            for n in self.out_layer:
                y.append(n.feed(res3))
            y_pred.append(y)
        return y_pred

    def train_grad_full_batch(self, data, y_trues, speed, epochs):
        data_with_bias = add_bias_to_data(data)
        y_trues_arr = y_to_arrays(y_trues)
        errs = []
        epchs = []
        for epoch in range(epochs):
            if (epoch + 1) % (epochs // 20) == 0 or epoch == 0:
                y_preds = self.predict(data)
                err = err_los_mc(y_trues_arr, y_preds)
                acc = err_acc(y_trues, arrays_to_y(y_preds), 0.1)
                epchs.append(epoch)
                errs.append(err)
                print("Epoch %d err: %.3f" % (epoch, err))
                print("Epoch %d acc: %.3f" % (epoch, acc))
                if acc > 0.99 and err < 0.01:
                    break
            for x, y_true in zip(data_with_bias, y_trues_arr):
                self.grad_iteration(x, y_true, speed)
        return epchs, errs

    def train_grad_stochastic(self, data, y_trues, speed, epochs):
        data_with_bias = add_bias_to_data(data)
        y_trues_arr = y_to_arrays(y_trues)
        errs = []
        epchs = []
        for epoch in range(epochs + 1):
            if (epoch + 1) % (epochs // 20) == 0 or epoch == 0:
                y_preds = self.predict(data)
                err = err_los_mc(y_trues_arr, y_preds)
                epchs.append(epoch)
                errs.append(err)
                acc = err_acc(y_trues, arrays_to_y(y_preds), 0.1)
                print("Epoch %d err: %.3f" % (epoch, err))
                print("Epoch %d acc: %.3f" % (epoch, acc))
                if acc > 0.99 and err < 0.01:
                    break
            n_rule = np.random.randint(0, len(data_with_bias))
            x = data_with_bias[n_rule]
            y_true = y_trues_arr[n_rule]
            self.grad_iteration(x, y_true, speed)
        return epchs, errs

    def grad_iteration(self, x, y_true, speed):
        res1 = []
        for n in self.NeuronsL1:
            res1.append(n.feed(x))
        res1.append(1)
        res2 = []
        for n in self.NeuronsL2:
            res2.append(n.feed(res1))
        res2.append(1)
        res3 = []
        for n in self.NeuronsL3:
            res3.append(n.feed(res2))
        res3.append(1)
        out = []
        for n in self.out_layer:
            out.append(n.feed(res3))

        # Изменение весов на выходном слое
        grad = []
        for j in range(len(out)):
            delta = out[j] - y_true[j]
            grad.append(delta * fun_der_sigm(out[j]))
            for i in range(len(res3)):
                self.out_layer[j].weights[i] = self.out_layer[j].weights[i] - speed * grad[j] * res3[i]

        # Изменение весов на третьем слое
        grad2 = []
        for i in range(len(res3)):
            grad_sum = 0
            for j in range(len(self.out_layer)):
                grad_sum += grad[j] * self.out_layer[j].weights[i] * fun_der_sigm(res3[i])
            grad2.append(grad_sum)
        for i in range(len(self.NeuronsL3)):
            for j in range(len(res2)):
                self.NeuronsL3[i].weights[j] = self.NeuronsL3[i].weights[j] - speed * grad2[i] * res2[j]

        # Изменение весов на втором слое
        grad3 = []
        for i in range(len(res2)):
            grad_sum = 0
            for j in range(len(self.NeuronsL3)):
                grad_sum += grad2[j] * self.NeuronsL3[j].weights[i] * fun_der_sigm(res2[i])
            grad3.append(grad_sum)
        for i in range(len(self.NeuronsL2)):
            for j in range(len(res1)):
                self.NeuronsL2[i].weights[j] = self.NeuronsL2[i].weights[j] - speed * grad3[i] * res1[j]

        # Изменение весов на первом слое
        grad4 = []
        for i in range(len(res1)):
            grad_sum = 0
            for j in range(len(self.NeuronsL2)):
                grad_sum += grad3[j] * self.NeuronsL2[j].weights[i] * fun_der_sigm(res1[i])
            grad4.append(grad_sum)
        for i in range(len(self.NeuronsL1)):
            for j in range(len(x)):
                self.NeuronsL1[i].weights[j] = self.NeuronsL1[i].weights[j] - speed * grad4[i] * x[j]

def get_res(y):
  if y > 0.5:
    return 1
  return 0

def get_res_multy(y):
  return np.argmax(y)