from packages import *
from load_data import load_cifar10_1d, load_mnist_1d, load_notmnist, load_yelp


class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

class NeuralUCBDiag:
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
            #print("fx:", fx)
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)
           # print(self.lamdba)
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
    parser = argparse.ArgumentParser(description='NeuralUCB')
    parser.add_argument('--dataset', default='cifar10', type=str, help='mnist, cifar10, notmnist, yelp')
    parser.add_argument('--hidden', default=100, type=int, help='Number of hidden layers for NN, default=100')
    parser.add_argument('--lam', default=0.1, type=float, help='Lambda Hyper-parameter, default=0.1')
    parser.add_argument('--nu', default=0.01, type=float, help='Nu Hyper-parameter, default=0.01')
    args = parser.parse_args()
    
    arg_size = 1
    arg_shuffle = 1
    arg_seed = 0
    arg_nu = args.nu
    arg_lambda = args.lam
    arg_hidden = args.hidden

    if args.dataset == "mnist":
        b = load_mnist_1d()
    elif args.dataset == "cifar10":
        b = load_cifar10_1d()
    elif args.dataset == "yelp":
        b = load_yelp()
    elif args.dataset == "notmnist":
        b = load_notmnist()
    elif args.dataset == "cifar10-mae":
        b = load_cifar10_mae()
        
    summ = 0
    regrets = []
    l = NeuralUCBDiag(b.dim, arg_lambda, arg_nu, arg_hidden)
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
    
    
    
    
    
