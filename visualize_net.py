from torch_model_rnn import binaryClassification
import torch
from torchviz import make_dot
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fpath = './models/neural-FNN-{}.pt'.format(1000)
model = torch.load(fpath)
model = model.to(device)

summary(model, (64,4))

make_dot(model)