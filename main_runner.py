import Trainer
import torch
from torch import nn
from torch.utils.data import DataLoader


model1 = Trainer.My_Model()

training_dataloader = DataLoader(dataset,batch_size=64)
testing_dataloader = DataLoader(dataset,batch_size=64)

epochs = 10

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    model1.trainer(training_dataloader)
    model1.tester(testing_dataloader)

print("Done!")


#create dataset
#make those changes in the code
#complte and run it
