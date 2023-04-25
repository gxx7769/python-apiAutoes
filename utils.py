import numpy as np 
import gym 
import torch 
import random
from argparse import ArgumentParser 
import os 
import pandas as pd
import itertools
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d
from collections import Counter

def arguments(): 

    parser = ArgumentParser()
    parser.add_argument('--env', default = 'Multi_Agent_Bs')

    return parser.parse_args()


def save(agents, rewards, args, r1, r2, d1, d2, d3, l1_req, l2_req, l3_req, l1_local, l2_local, l3_local, l1_coop, l2_coop, l3_coop, l1_server, l2_server, l3_server):
# def save(agents, rewards, args):

    path = './runs/{}/'.format(args.env)
    try: 
        os.makedirs(path)
    except: 
        pass 
    for agent in agents:
        torch.save(agent.q.state_dict(), os.path.join(path, 'model_state_dict_{}'.format(agents.index(agent))))

    plt.cla()
    plt.plot(rewards, c = 'r', alpha = 0.3)
    plt.plot(gaussian_filter1d(rewards, sigma = 5), c = 'r', label = 'Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Multi-Branching DDQN: {}'.format(args.env))
    plt.savefig(os.path.join(path, 'reward.png'))
    pd.DataFrame(list(zip(rewards, r1, r2, d1, d2, d3, l1_req, l2_req, l3_req, l1_local, l2_local, l3_local, l1_coop, l2_coop, l3_coop, l1_server, l2_server, l3_server)), columns = ['Reward','r1','r2','d1','d2','d3','l1_req', 'l2_req', 'l3_req', 'l1_local_hit', 'l2_local_hit', 'l3_local_hit', 'l1_coop_hit', 'l2_coop_hit', 'l3_coop_hit', 'l1_server_hit', 'l2_server_hit', 'l3_server_hit' ]).to_csv(os.path.join(path, 'rewards.csv'), index = False)
    # pd.DataFrame(list(zip(rewards)), columns = ['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index = False)

class AgentConfig:

    def __init__(self, 
                 epsilon_start = 1.,
                 epsilon_final = 0.01,
                 epsilon_decay = 8000,
                 gamma = 0.99, 
                 lr = 5e-4,
                 target_net_update_freq = 1000, 
                 memory_size = 100000, 
                 batch_size = 128,
                 learning_starts = 2000,
                 max_frames = 100000):

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.epsilon_by_frame = lambda i: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * i / self.epsilon_decay)

        self.gamma =gamma
        self.lr =lr

        self.target_net_update_freq =target_net_update_freq
        self.memory_size =memory_size
        self.batch_size =batch_size

        self.learning_starts = learning_starts
        self.max_frames = max_frames


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = [] 
        dones = []

        for b in batch: 
            states.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])
            dones.append(b[4])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class BsEnv():

    def __init__(self, n_step):
        self.n_user = 3
        self.n_cache = 3
        self.n_file = 25
        self.n_layer = 3
        self.n_bs = 3

        self.s_dim = self.n_user

        self.a1_dim = self.n_cache
        self.a2_dim = 1
        self.a3_dim = 1
        self.a_dim = self.a1_dim+self.a2_dim+self.a3_dim

        self.a1_n = self.n_file
        # self.a1_n = self.n_file
        self.a2_n = self.n_layer ** self.n_user
        self.a3_n = self.n_layer ** self.n_cache

        self.file_set = np.arange(1, self.n_file+1)
        #self.file_set = np.array([a*10+np.arange(1, self.n_layer+1) for a in np.arange(1, self.n_file+1)]).ravel()
        self.q = [list(x) for x in itertools.product(np.arange(1, self.n_layer+1), repeat=self.n_user)]
        self.layer_set = [list(x) for x in itertools.product(np.arange(1, self.n_layer+1), repeat=self.n_cache)]

        self.popularity = self.zipf(self.n_file)
        self.s_all = self.reset(n_step)

    def zipf(self, n_file):

        denominator = 0
        alpha = 1.5
        for i in np.arange(1, n_file+1):
            denominator += 0.1 / pow(i, alpha)
            normalize = 1 / denominator
        popularity = (0.1 / pow(np.arange(1, n_file+1), alpha)) * normalize
        return popularity

    def reset(self, n_step):
        p1=[1/3, 1/3, 1/3]
        p2=[1/4, 1/4, 1/2]
        p3=[0.15,0.15,0.7]
        self.s_all = []
        for t in range(n_step+1):
            s_current = []
            for k in range(self.n_bs):
                user_request = list(np.random.choice(np.arange(1, self.n_file + 1), self.n_user, p=self.popularity)*10
                                + np.random.choice(np.arange(1, self.n_layer + 1), self.n_user,p=p2))
                s_current.append(user_request)
            #self.s_all.append(sum(s_current,[]))
            self.s_all.append(s_current)
        return self.s_all

    def process(self, s):

        return torch.tensor(s).float()
        # return torch.tensor(s).reshape(1,-1).float()

    def step(self, a_s, step):

        ns = np.array(self.s_all[step+1])
        done = False

        cac = []
        for n in range(self.n_bs):
            cac.append(np.array([self.file_set[a] for a in a_s[n][:self.n_cache]]) * 10 + np.array(self.layer_set[a_s[n][-1]]))

        del_q = []
        for n in range(self.n_bs):
            del_q.append(np.array(self.q[a_s[n][-2]]))

        #有传递质量
        delay = 0
        delay2 = 0
        delay3 = 0
        q_diff = 0
        #层特性参数
        l1_req_num=0
        l2_req_num=0
        l3_req_num=0
        l1_local_hit=0
        l2_local_hit = 0
        l3_local_hit = 0
        l1_coop_hit=0
        l2_coop_hit = 0
        l3_coop_hit = 0
        l1_server_hit=0
        l2_server_hit = 0
        l3_server_hit = 0
        for n in range(self.n_bs):
            for u in range(self.n_user):
                q = ns[n][u] % 10
                needed = [ns[n][u]//10*10 + l for l in np.arange(1, q+1)]
                for need in needed:
                    if need % 10 == 1:
                        l1_req_num+=1
                        if need in cac[n]:
                            l1_local_hit+=1
                        elif need in np.delete(cac, n, axis=0):
                            l1_coop_hit+=1
                        else:
                            l1_server_hit+=1
                    if need % 10 == 2:
                        l2_req_num+=1
                        if need in cac[n]:
                            l2_local_hit+=1
                        elif need in np.delete(cac, n, axis=0):
                            l2_coop_hit+=1
                        else:
                            l2_server_hit+=1
                    if need % 10 == 3:
                        l3_req_num+=1
                        if need in cac[n]:
                            l3_local_hit+=1
                        elif need in np.delete(cac, n, axis=0):
                            l3_coop_hit+=1
                        else:
                            l3_server_hit+=1


        #计算时延
        for n in range(self.n_bs):
            for u in range(self.n_user):
                q_ = del_q[n][u]
                q = ns[n][u] % 10
                q_diff -= abs(q - q_)
                needed = [ns[n][u]//10*10 + l for l in np.arange(1, q_+1)]
                for need in needed:
                    if need % 10 == 1:
                        if need in cac[n]:
                            delay -= 0
                        elif need in np.delete(cac, n, axis=0):
                            delay -= 0.3
                        else:
                            delay -= 2
                    if need % 10 == 2:
                        if need in cac[n]:
                            delay2 -= 0
                        elif need in np.delete(cac, n, axis=0):
                            delay2 -= 0.3
                        else:
                            delay2 -= 2
                    if need % 10 == 3:
                        if need in cac[n]:
                            delay3 -= 0
                        elif need in np.delete(cac, n, axis=0):
                            delay3 -= 0.3
                        else:
                            delay3 -= 2

        del_q = sum(list(map(lambda x: list(x), del_q)), [])
        c=del_q.count(3)
        b=del_q.count(2)+c

        r1 = q_diff/((self.n_layer-1)*(self.n_bs*self.n_user))
        r2 = (delay+delay2+delay3)/((np.sum(del_q))*2)
        delay  = (delay / 18)
        delay2 = (delay2 / (2*b)) if b!=0 else 0
        delay3 = (delay3 / (2*c)) if c!=0 else 0
        r=0.3*r1+0.7*r2

        return self.process(ns), r, done, r1, r2, delay, delay2, delay3, \
        l1_req_num, l2_req_num,l3_req_num,l1_local_hit, l2_local_hit, l3_local_hit,\
               l1_coop_hit, l2_coop_hit, l3_coop_hit, l1_server_hit, l2_server_hit, l3_server_hit,
