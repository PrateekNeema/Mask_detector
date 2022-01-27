import neural_network
import torch
from torch import nn
from torch.utils.data import DataLoader



class My_Model():

    def __init__(self):

        self.learning_rate = 1e-3  #########
        self.batch_size = 64  #########
        self.epochs = 5  #######

        #initialize model
        self.neural_net = neural_network.NeuralNetwork()

        # Initialize the loss function
        self.loss_fn = nn.NLLLoss()
        #(Negative Log Likelihood) for classification or #loss_fn = nn.CrossEntropyLoss() #combines nn.LogSoftmax and nn.NLLLoss.

        #Initialize stochastic gradient descent optimizer
        self.optimizer = torch.optim.SGD(neural_net.parameters(), lr=learning_rate)




    def learn(self,batch,(X,y)):

        # Compute prediction and loss
        prediction = self.neural_net(X)
        self.loss_fn(prediction, y)

        # Backpropagation
        self.optimizer.zero_grad()
        self.loss_fn.backward()
        self.optimizer.step()


    def trainer(self,training_data_dataloader):

        for batch,(X,y) in enumerate(training_data_dataloader):
            self.learn(batch,(X,y))


    def tester(self,testing_data_dataloader):

        size = len(testing_data_dataloader.dataset)
        num_batches = len(testing_data_dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in testing_data_dataloader:
                pred = self.neural_net(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")













