#Однослойные нейронные сети
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl # Импорт билиотек

# путь
input_data = np.loadtxt("/qqq/Pr/neural_simple.txt")

# Превращаем таблицу в 2 столбца с 2 метками
data = input_data[:, 0:2]
labels = input_data[:, 2:]

# График ввода данных
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data') 

# установка min и max измерения
dim1_min, dim1_max = data[:,0].min(), data[:,0].max()
dim2_min, dim2_max = data[:,1].min(), data[:,1].max()

# Число нейронов
nn_output_layer = labels.shape[1]

# Делаем однослойную сеть
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
neural_net = nl.net.newp([dim1, dim2], nn_output_layer)

# Задаем эпохи и скорость тренировки 
error = neural_net.train(data, labels, epochs = 200, show = 20, lr = 0.01)

# Визуализация графика процесса тренеровки
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()
plt.show()

print('\nTest Results:')
data_test = [[1.5, 3.2], [3.6, 1.7], [3.6, 5.7],[1.6, 3.9]] 
for item in data_test:
    print(item, '-->', neural_net.sim([item])[0])
