import time
import timeit
from multiprocessing import Pool
import multiprocessing as mp
import os

import torch

from Lean4Gym import Lean4Gym, ProofState
# print(torch.__version__)
from torch.nn.parallel import DataParallel  
# torch.cuda.set_device(2)

# import lean_dojo
# from lean_dojo import *
from model import policy_model
from model import value_model
from trainer import Trainer
from mcts import Node
from mcts import MCTS
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 


args = {
    
    'batch_size': 4,
    'numIters': 10,                                # Total number of training iterations
    'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 20,                                  # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 10,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth',                 # location to save latest set of weights
    'TACRIC_NUMBER': 8,
    'feature_size':100
    # 'MAX_ROUND_NUMBER' : 10
}


def list_files(directory):
    filelist = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            print(file)
            filelist.append(file)
    return filelist


state_list = []
lean_list = []

feature_size = args['feature_size']  # 特征向量的size
time_out = 120


device_ids = list(range(torch.cuda.device_count()))  
policyModel = policy_model(feature_size*2, device).to(device)
valueModel = value_model(feature_size, device).to(device)
print("hello,开始了！！")


checkpoint_policy = torch.load("/home/wanglei/AAAI/lean_ATG/leanproject/ATG/policy_model")
policyModel.load_state_dict(checkpoint_policy['state_dict'])

checkpoint_value = torch.load("/home/wanglei/AAAI/lean_ATG/leanproject/ATG/value_model")
valueModel.load_state_dict(checkpoint_value['state_dict'])


count = 0
new_theorems = []

#待证明策略：
lean_dir = "/home/wanglei/Project/testfolder/succ"
# lean_dir = "/home2/wanglei/Project/testfolder"
file_list = list_files(lean_dir)
# print(len(file_list))

lean_workdir = "/home/wanglei/AAAI/lean_ATG/leanproject" # Lean工程的根目录


def long_time_task(name):
    print(new)


if __name__ == '__main__':
    print(mp.cpu_count())
    pool = Pool(processes=mp.cpu_count() // 2)
    t1 = timeit.default_timer()
    res = pool.map(long_time_task, s)
    t2 = timeit.default_timer()
    print(t2 - t1)
    pool.close()
    pool.join()
    print(res)

    t3 = timeit.default_timer()
    for i in range(10):
        long_time_task(s[i])
    t4 = timeit.default_timer()
    print(t4 - t3)
