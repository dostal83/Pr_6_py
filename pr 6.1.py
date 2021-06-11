#Классификатор на основе перцептрона
import matplotlib.pyplot as plt
import neurolab as nl # импорт библиотек

# Значения ввода (контр. об) 
input = [[0, 0], [0, 1], [1, 0], [1, 1]]
target = [[0], [0], [0], [1]]

# Создаем нейронку 
net = nl.net.newp([[0,1], [0,1],], 1) # 2 входа на 1 нейрон

# Тренеруем сеть по принципу delta
error_progress = net.train(input, target, epochs=100, show=10, lr=0.1) 

# Визуалка
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.grid()
plt.show()
