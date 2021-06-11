import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

input_data = np.loadtxt("/Users/Elian/neural_simple.txt")

data = input_data[:, 0:2]
labels = input_data[:, 2:]

plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data') 

dim1_min, dim1_max = data[:,0].min(), data[:,0].max()
dim2_min, dim2_max = data[:,1].min(), data[:,1].max()

nn_output_layer = labels.shape[1]

dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
neural_net = nl.net.newp([dim1, dim2], nn_output_layer)

error = neural_net.train(data, labels, epochs = 200, show = 20, lr = 0.01)

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