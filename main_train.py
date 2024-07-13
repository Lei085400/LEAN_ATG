import os

import torch
from Lean4Gym import Lean4Gym, ProofState
from model import policy_model
from model import value_model
from trainer import Trainer
from mcts import Node
from mcts import MCTS

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 

args = {
    'batch_size': 64,
    'numIters': 20,                                # Total number of training iterations
    'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 5,                                  # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 1000,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth',                 # location to save latest set of weights
    'TACRIC_NUMBER': 8,
    'feature_size':100,
    'max_count': 40
    # 'MAX_ROUND_NUMBER' : 10
}


policyModel = policy_model(args['feature_size']*2, device).to(device)
valueModel = value_model(args['feature_size'], device).to(device)

state_list = []
lean_list = []

time_out = 600

def list_files(directory):
    filelist = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            print(file)
            filelist.append(file)
    return filelist


#待证明策略：
lean_dir = "/home2/wanglei/Project/testfolder/succ"
# lean_dir = "/home2/wanglei/Project/testfolder"
file_list = list_files(lean_dir)
# print(len(file_list))

lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
for i, file in enumerate(file_list):
    print("============================================")
    lean_file = "testfolder/succ/" + file  # 待证明定理的Lean文件
   
    print("证明定理为:{}".format(file))
    lean = Lean4Gym(lean_workdir, lean_file)
    try:
        state = lean.getInitState()
    except:
        print("状态异常")
        continue
    
    state_list.append(state)
    lean_list.append(lean)



trainer = Trainer(policyModel, valueModel, args, device)
print("马上开始训练")
trainer.learn(state_list, lean_list)

# trainer.train()

