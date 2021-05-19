import numpy as np


# network description:
#    * input layer -> 3 units
#    * 1 hidden layers -> 4 units
#    * 2 weight matrices -> (number of neuron of inputs * number of transfer layer) eg. in this case
#         todo: from input layer to hidden layer (3*4)
#         todo: from hidden to output (4*1)
#    * output layer -> 1 unit

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self,x,y,hidden_neuron):
        # AFH1 -> Activation function of hidden 1
        self.AFH1=[]
        self.input=x
        # print('Inputs \n',self.input)
        # print()
        self.weights1=np.random.rand(self.input.shape[1],hidden_neuron)
        # print('Weights matrix for hidden layer 1 \n',self.weights1)
        # print()
        self.weights2=np.random.rand(hidden_neuron,1)
        # print('Weights matrix for hidden layer 2 \n',self.weights2)
        # print()
        self.y=y
        # print('Actual Output \n',self.y)
        # print()
        self.output=np.zeros(self.y.shape)  # y hat
        # print('h(x) -> Output \n',self.output)
        # print()

    def feedforward(self,iter):
        self.AFH1=sigmoid(np.dot(self.input,self.weights1))
        # print(str(iter)+' => AFH1 \n',self.AFH1)
        # print()

        self.output=sigmoid(np.dot(self.AFH1,self.weights2))
        # print(str(iter)+' => h(x) of feedforward \n',self.output)
        # print()

    def backpropagation(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2=np.dot(self.AFH1.T,(2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        print('d_weights2  \n',d_weights2)
        print()

        d_weights1=np.dot(self.input.T,
                          (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                  self.weights2.T) * sigmoid_derivative(self.AFH1)))
        print('d_weights1 \n',d_weights1)
        print()

        # update the weights with the derivative (slope) of the loss function
        self.weights1+=d_weights1
        self.weights2+=d_weights2
