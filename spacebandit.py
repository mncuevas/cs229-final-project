'''
Adapted from Neural Bandits with UCB Expolartion
'''

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

device = torch.device('cuda')

def main(args):
    arg_nu = args.nu
    arg_lambda = args.lam
    arg_hidden = args.hidden

    b = load_cifar10_mae()
        
    summ = 0
    regrets = []
    l = SpaceBandit(b.dim, arg_lambda, arg_nu, arg_hidden)
    for t in range(10000):
        context, rwd = b.step()
        arm_select, f_res, ucb = l.select(context)
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        summ+=reg
        l.update(context[arm_select], r)
        if t<2000:
            if t%10 == 0:
                #print(rwd)
                loss = l.train()
        else:
            if t%100 == 0:
                loss = l.train()
        regrets.append(summ)
        if t % 50 == 0:
            print('{}: {:}, {:.3}, {:.3e}'.format(t, summ, summ/(t+1), loss))
            
    print("round:", t, summ)

class load_cifar10_mae:
    def __init__(self, is_shuffle=True):
        #Fetch data
        batch_size = 1
        x_train = torch.load('./data/cifar10_embeddings.pt')
        y_train = torch.load('./data/cifar10_labels.pt')
        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=is_shuffle, num_workers=0)
        self.dataiter = iter(trainloader)
        self.n_arm = 3
        self.dim = 192 * 3

    def step(self):
        x, y = self.dataiter.next()
        x = x.cpu()
        d = x.numpy()[0]
        target = int(y.item()/4.0)
        X_n = []
        for i in range(3):
            front = np.zeros((192*i))
            back = np.zeros((192*(2 - i)))
            new_d = np.concatenate((front,  d, back), axis=0)
            X_n.append(new_d)
        X_n = np.array(X_n)
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
        return X_n, rwd



class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.activate(x)
        x =self.fc2(x)
        return x

class SpaceBandit:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100):
        self.func = Network(dim, hidden_size=hidden).to(device)
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).to(device)
        self.nu = nu

    def select(self, context):
        tensor = torch.from_numpy(context).float().to(device)
        mu = self.func(tensor)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        f_res = []
        ucb = []
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))
            f_res.append(fx.item())
            ucb.append(sigma.item())
            sample_r = fx.item() + sigma.item()
            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm, f_res, ucb
    
    def update(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)

    def train(self):
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                delta = self.func(c.to(device)) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpaceBandit')
    parser.add_argument('--hidden', default=12500, type=int, help='Number of hidden layers for NN')
    parser.add_argument('--lam', default=0.0001, type=float, help='Lambda Hyper-parameter')
    parser.add_argument('--nu', default=0.01, type=float, help='Nu Hyper-parameter')
    args = parser.parse_args()
    
    main(args)