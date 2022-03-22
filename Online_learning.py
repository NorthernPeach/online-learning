def data_prep(file_name):
    data = []
    with open(file_name) as file:
        for line in file:
            data.append(line.strip())
    return data

def count(item):
    ones = 0
    for digit in item:
        if digit == '1':
            ones += 1
    return ones

def factorial(x):
    if x > 2:
        return x * factorial(x - 1)
    else:
        return 2

def online_learning(file_name, a, b):
    data = data_prep(file_name)
    for i, item in enumerate(data):
        N = int(len(item))
        m = count(item)
        f = factorial(N) / (factorial(m) * factorial(N-m))
        likelihood = ((m / N)**m) * ((1 - m / N)**(N - m)) * f
        print('Case {}: '.format(i+1), item, '\nLikelihood: ', likelihood)
        print('Beta prior:      a = {0}, b = {1}'.format(a, b))
        a += m
        b += N - m
        print('Beta posterior:  a = {0}, b = {1}'.format(a, b))
        print('')

path = 'C:/Users/Khristina/PycharmProjects/ML/test.txt'
online_learning(path, 10, 1)