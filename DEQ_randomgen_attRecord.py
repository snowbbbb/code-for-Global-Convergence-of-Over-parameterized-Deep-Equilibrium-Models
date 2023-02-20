import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import os
import torch.nn.functional as F
import random
from torch.utils.data import TensorDataset
import torch.autograd as autograd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
SEED = 2
torch.manual_seed(SEED)
random.seed(SEED)　　 
torch.cuda.manual_seed(SEED)　  
torch.backends.cudnn.deterministic = True
'''
########################################################################
# Download and define the training set.

samplesize = 10000
netDepth = 2
print('netdeptg:',netDepth)
trainTransform = transforms.Compose([transforms.Grayscale(1),
                                transforms.ToTensor()])



matrix=np.random.rand(1000,1000)
print("matrix is",np.size(matrix),matrix)
U,s,V=np.linalg.svd(matrix)
print('U is',np.size(U),U,type(U))
allData=torch.eye(1000)
allLabel =torch.rand(1000)
print(allLabel)

train_ids = TensorDataset(allData, allLabel)
trainloader = torch.utils.data.DataLoader(dataset=train_ids, batch_size=len(train_ids), shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_size=1



def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, D = x0.shape
    X = torch.zeros(bsz, m, D, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, D, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:, k % m].view_as(x0), res


class Net(nn.Module):
    def __init__(self, width,Wstd):
        super(Net, self).__init__()
        self.wmatrix=nn.Linear(width,width)
        torch.nn.init.normal_(self.wmatrix.weight, mean=0, std=Wstd)
    def forward(self, z, x):
        y = F.relu(self.wmatrix(z)+x) * (math.sqrt(1 / width))
        return y

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g

        z.register_hook(backward_hook)
        return z

def get_jacobian(model, x):
    nc = x.size()[0]
    ny = x.size()[2]
    nx = x.size()[1]
    noutputs = 10
    x = x.reshape(nc * nx * ny)
    x = x.repeat(noutputs, 1)
    x.requires_grad_(True)
    y = model(x.reshape(noutputs, nc, nx, ny))
    y.backward(torch.eye(noutputs).cuda())
    return x.grad.data



def setup_and_train(epochs, lr, width, Wstd):
    ######################################################################
    # Model setup
    f = Net(width, Wstd)
    linear_input = nn.Linear(1000, width)
    linear_output = nn.Linear(width, output_size)
    torch.nn.init.normal_(linear_input.weight, mean=0, std=1)
    torch.nn.init.normal_(linear_output.weight, mean=0, std=1)
    model = nn.Sequential(nn.Flatten(),
                          linear_input,
                          DEQFixedPoint(f, anderson, tol=1e-2, max_iter=25),
                          linear_output).to(device)
    model.cuda();
    newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    ######################################################################
    # Define criterion and optimizer
    criterion = torch.nn.MSELoss(reduce=True, size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
   # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=1.0)
  
    ###################################2###################################
    # Train the model
    model.train()
    fid = open('DEQ_random_' + str(width) + '_' + str(Wstd) + '_ExperimentResult.txt', 'w')
   # fid2 = open('DEQ_random_' + str(width) + '_' + str(Wstd) +  '_max_singularvalue_ExperimentResult.txt', 'w')
    loss_plot=[]

    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_start_time = time.time()
        running_loss = 0.0
        log_interval = 100
        train_acc = 0.0
        for batch, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(), data[1].cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if hasattr(torch.cuda, 'empty_cache'):
	            torch.cuda.empty_cache()
            outputs = model(inputs)

            outVector=newmodel(inputs)
            if width <1000:
              matrixA = torch.mm(outVector.T, outVector)
            else:
              matrixA = torch.mm(outVector, outVector.T)
            (lambdamin,lambdaVec) = torch.eig(matrixA)
            print("lambda_min", lambdamin[-1])
            WMatrix = f.wmatrix.weight * (math.sqrt(1 / width))
            matrixA = torch.mm(WMatrix.T, WMatrix)
            (singmax, singularVec) = torch.eig(matrixA)
            print("singular_max", torch.sqrt(singmax[0]))

            loss= criterion(outputs.reshape(labels.shape),labels.float())
            #loss = criterion(outputs, labels)
            #train_correct = (pred == labels).sum()
            #train_acc += train_correct.item()
            if hasattr(torch.cuda, 'empty_cache'):
	            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
        cur_loss = running_loss / (batch + 1)
        print('| end of epoch {:3d} | time / epoch {:5.2f}s | loss {:5.2f} '.format
              (epoch + 1, (time.time() - epoch_start_time), cur_loss))
        loss_plot.append(cur_loss)
        fid.write(str(cur_loss)+'\n')
        #fid1.write(str(torch.norm(lambdamin[-1], p=2).item()) + '\n')
       # fid2.write(str(torch.norm(torch.sqrt(singmax[0]), p=2).item())+'\n')   
        running_loss = 0.

    plt.figure(1)
    plt.plot(np.linspace(1, epochs, len(loss_plot)), loss_plot)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('L2loss.png')


epochs =10000      # Number of epochs
Wstd = 0.4
width = 256        # Width
lr = 0.000001

setup_and_train(epochs, lr, 7000, Wstd)
