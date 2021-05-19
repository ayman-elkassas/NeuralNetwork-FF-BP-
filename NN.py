import numpy as np
from NeuralNetwork import *

# input sample as feature array
X=np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
])

y=np.array([
    [0],
    [1],
    [1],
    [0],
])

nn=NeuralNetwork(X,y,hidden_neuron=4)

for i in range(500):
    nn.feedforward(i+1)
    nn.backpropagation()
    print('--------------------------------')

print(nn.output)
