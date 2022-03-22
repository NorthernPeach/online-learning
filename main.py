import os
import gzip
import numpy as np
import math


path = 'C:/Users/Khristina/PycharmProjects/ML/MNIST/'
files = os.listdir(path)
test_img, test_labels, train_img, train_labels = ([gzip.open(path+'{}'.format(file), 'rb').read() for file in files])

# --------- Data preprocessing ---------------
magic, n_img, rows, cols = [int.from_bytes(train_img[i*4:(i+1)*4], byteorder='big') for i in range(0, 4)]
unsigned_train_img = np.asarray(list(train_img[16:]))
processed_data = []

for i in range(0, len(unsigned_train_img), 784):
    a = np.asarray(unsigned_train_img[i:i + 784])
    processed_data.append(a)

test_unsigned_img = np.asarray(list(test_img[16:]))
test_data = []
for i in range(0, len(test_unsigned_img), 784):
    a = np.asarray(test_unsigned_img[i:i + 784])
    test_data.append(a)

train_labels = np.asarray(list(train_labels[8:]))
test_labels = np.asarray(list(test_labels[8:]))
print('train_labels: {0}\ntest labels length: {1}'.format(len(train_labels), len(test_labels)))

# -------- 1. Naive Bayes ----------------------

def make_LUT(processed_data, train_labels):
    # create a matrix to store data
    LUT = np.zeros((10, 784, 32), dtype=int)
    prior = np.zeros(10, dtype=int)

    for image, label in zip(processed_data, train_labels):
        prior[label] += 1
        #image = map32(image)
        for position in range(784):
            value = image[position]
            LUT[label][position][int(value/8)] += 1

    prior = prior/len(train_labels)
    return LUT, prior

def predict(probability, gRtr):
    print('Postirior (in log scale): ')
    for label in range(10):
        print(label, ': ', probability[label])
    prediction = np.argmin(probability)
    print('Prediction: ', prediction, ', Ans: ', gRtr)
    print('')
    if prediction == gRtr:
        return 0
    else:
        return 1

def print_image(LUT):
    print('Imagination of numbers in Bayesian classifier:')
    print('')
    for label in range(10):
        print(i, ':')
        for j in range(28):
            for k in range(28):
                temp = 0
                for t in range(16):
                    temp += LUT[label][j * 28 + k][t]
                for t in range(16, 32):
                    temp -= LUT[label][j * 28 + k][t]
                if temp > 0:
                    print('0', end = ' ')
                else:
                    print('1', end = ' ')
            print('')
        print('')

def discrete_Bayes(processed_data, train_labels, test_data, test_labels):
    LUT, prior = make_LUT(processed_data, train_labels)
    error = 0
    for test_image, gRtr in zip(test_data, test_labels):
        posterior = np.zeros(10, dtype=float)
        for label in range(10):
            posterior[label] += np.log(prior[label])

            for position in range(784):
                value = LUT[label][position][int(test_image[position]/8)]
                if value == 0:
                    posterior[label] += np.log(1e-3/sum(LUT[label][position][:]))
                else:
                    posterior[label] += np.log(LUT[label][position][int(test_image[position]/8)]/sum(LUT[label][position][:]))
        posterior = posterior/sum(posterior)   # normalize posterior for this label
                                               # for a particular image
        error += predict(posterior, gRtr)
    print_image(LUT)
    print('Error rate: ', error/10000)

# ----------Part for the continuous mode-------------------
def mean_var(processed_data, train_labels):
    mean = np.zeros((10, 784), dtype=float)
    square = np.zeros((10, 784), dtype=float)
    var = np.zeros((10, 784), dtype=float)
    prior = np.zeros(10)

    for image, label in zip(processed_data, train_labels):
        prior[label] += 1
        for position in range(784):
            mean[label][position] += image[position]
            square[label][position] += (image[position])**2

    for label in range(10):
        for position in range(784):
            mean[label][position] = mean[label][position] / prior[label]
            square[label][position] = square[label][position] / prior[label]
            var[label][position] = square[label][position] - (mean[label][position])**2
            var[label][position] = np.where(var[label][position] == 0, 1e-3, var[label][position])

    prior = prior/len(train_labels)
    return mean, var, prior

def print_contin_image(mean):
    print('Imagination of numbers in Bayesian classifier:')
    print('')
    for i in range(10):
        print(i, ':')
        for j in range(28):
            for k in range(28):
                if mean[i][j * 28 + k] < 128:
                    print('0', end = ' ')
                else:
                    print('1', end = ' ')
            print('')
        print('')

def Gaussian(x, mean, var):
    coef = np.log(1.0 / math.sqrt(2 * math.pi * var))
    exp = (x - mean)**2 / (2 * var)
    res = coef - exp
    return res

def contin_Bayes(processed_data, train_labels, test_data, test_labels):
    mean, var, prior = mean_var(processed_data, train_labels)
    error = 0
    for image, grTr in zip(test_data, test_labels):
        posterior = np.zeros(10, dtype=float)
        for label in range(10):
            posterior[label] += np.log(prior[label])

            for position in range(784):
                posterior[label] += Gaussian(image[position], mean[label][position], var[label][position])
        posterior = posterior/sum(posterior)    # normalize probability
        error += predict(posterior, grTr)
    print_contin_image(mean)
    print("Error rate: ", error/10000)

# Please, write the toggle option 0 or 1 here
toggle_option = 0

if toggle_option == 0:
    discrete_Bayes(processed_data, train_labels, test_data, test_labels)
elif toggle_option == 1:
    contin_Bayes(processed_data, train_labels, test_data, test_labels)