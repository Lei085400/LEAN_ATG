#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Paper from http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf .
import os
import sys
import math
import random
import numpy as np
# from lean_dojo import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import json
import heapq
import transformers
import subprocess
import time
from datetime import datetime
from tqdm import tqdm, trange
from pathlib import Path
# from lean_dojo import *
import traceback
import copy
import vllm
from datetime import datetime
from tqdm import tqdm, trange
from pathlib import Path
from Lean4Gym import *
# from lean_dojo import *
import traceback
# TACRIC_NUMBER = 8
MAX_ROUND_NUMBER = 10
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

model_name_or_path = "/home/wanglei/AAAI/lean_ATG/model/pythia2.8b_choose"

model = vllm.LLM(
    model=model_name_or_path,
    tensor_parallel_size=1,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
    dtype='float16'
)
tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(model_name_or_path)

# 模型的prompt输入
def _prompt_proofstep(ts):
    prompt = f"[GOAL]{ts}[PROOFSTEP]"
    if(len(prompt)>2048):
      prompt = prompt[:2048]
    return prompt

# 去重排序
def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_

# 模型输入产生输出
def generate_vllm(prompt, model, tokenizer, temperatures, num_samples, stop, max_tokens=1024):
    texts, scores = [], []
    for temperature in temperatures:
        params = vllm.SamplingParams(
            n=num_samples,
            temperature=temperature,
            use_beam_search=temperature==0.0,
            max_tokens=max_tokens,
            stop=stop,
            logprobs=0,
            top_k=-1
        )
        outputs = model.generate([prompt], params, use_tqdm=False)

        # for tac in outputs[0].outputs:
        #     print(tac)
        

        if len(outputs) == 0:
            return [], []
        for output in outputs[0].outputs:
            text = output.text.replace(tokenizer.eos_token, '') 
            score = output.cumulative_logprob/max(len(output.token_ids), 1)
            texts.append(text)
            scores.append(score)

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores

def encode_state(state, feature_size):
    # state = str([str(sublist) for sublist in state])
    if(state is None):
        state = "None"
    encode_state = [ord(char) for char in state]
    if(len(encode_state)<=feature_size):
        encode_state += [0]*(feature_size-len(encode_state))  #list
    else:
        encode_state = encode_state[:feature_size]
    # print("encode")
    # print(encode_state)
    return encode_state
 
def encode_tactic(tactic,feature_size):
    # tactic = str([str(sublist) for sublist in tactic])
    encode_tactic = [ord(char) for char in tactic]
    if(len(encode_tactic)<=feature_size):
        encode_tactic += [0]*(feature_size-len(encode_tactic))
    else:
        encode_tactic = encode_tactic[:feature_size]
    return encode_tactic


def cosine_similarity(vector1, vector2):
    # 计算向量的内积
    dot_product = np.dot(vector1, vector2)
    
    # 计算向量的长度
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    # 计算余弦相似度
    if norm_vector1 != 0 and norm_vector2 != 0:
        cosine_sim = dot_product / (norm_vector1 * norm_vector2)
    else:
        cosine_sim = 0  # 避免除零错误
    
    return cosine_sim


def tactic_generator(state):

  state = state.getTacticState()
  tactic_candidates, scores = generate_vllm(_prompt_proofstep(state), model, tokenizer, 
                              temperatures=[0], num_samples=8, stop=tokenizer.eos_token)
      
  return tactic_candidates

def generate_theorem(node):
  return node.state.tacticState
 
#判断是否为新定理
def new_theorem(node):
  return node.flag

class Node(object):
  """
  蒙特卡罗树搜索的树结构的Node，包含了父节点和直接点等信息，还有用于计算UCB的遍历次数和quality值，还有游戏选择这个Node的State。
  """

  def __init__(self,state):
    self.parent = None
    self.children = []
    self.prob = 0
    self.puct = 0
    self.visit_times = 0
    self.quality_value = 0.0
    self.flag = True #记录节点是否可行，不可行则为Flase
    self.new = False #记录该节点是否产生了新定理
    self.tac = None  #记录当前状态是通过哪个策略得到的
    self.state = state
    self.path = None #新定理生成策略路径
    self.tactic_candidates = None    
    self.assersion = None
    self.depth = 0 
    self.similarity = 0
    self.llmflag = True #代表llm能产生有用的策略

  def set_state(self, state):
    self.state = state

  def get_state(self):  
    return self.state

  def set_parent(self, parent):  
    self.parent = parent

  def get_children(self):  
    return self.children

  def get_visit_times(self):  
    return self.visit_times

  def visit_times_add_one(self):  
    self.visit_times += 1

  def get_quality_value(self): 
    return self.quality_value

  def quality_value_add_n(self, n):  
    self.quality_value += n

  def is_all_expand(self,TACRIC_NUMBER): #### 判断该状态下，是否所有的list中的策略都尝试过了
    return len(self.children) == TACRIC_NUMBER
  
  def add_child(self, sub_node):
    sub_node.set_parent(self)
    self.children.append(sub_node)
  
  def select_action(self):
    """
    Select action according to the visit count distribution and the temperature.
    """
    visit_counts = np.array([child.visit_times for child in self.children])
    actions = [action for action in self.children.tac]
    action = actions[np.argmafx(visit_counts)]
    return action

  def is_terminal(self):  ########### 节点是否报错
    if(self.state.error is not None):
      self.flag == False 
      return True
    else:
      self.flag == True
      return False

  def compute_reward(self):   ############# 证明成功为1，失败为0
    if (self.flag == False):
      return -1
    elif (self.flag == True):
      if(self.new == True):
        return 1
      else:
        return 0
  
  def proof(self,state,tac,lean):
    try:
      result = lean.run_tactic(state, [tac])
    except:
      return state
    return result

  def get_next_state_with_random_choice(self, lean, index):  ############# 根据当前state输入大模型，获取策略list后，随机选择其中一个策略，返回执行该随机策略后的状态
    # if(self.state==[]):
    #     self.tactic_candidates = tactic_generator()
    # else:
    #     self.tactic_candidates = self.parent.tactic_candidates
    #     self.tactic_candidates.append("新定理")  
    
    # tactic_candidates = self.tactic_candidates
    # random_choice = random.choices([choice for choice in tactic_candidates],k=1)
    
    if(self.tactic_candidates is None):
      self.tactic_candidates = tactic_generator(self.state)
    tactic_candidates = self.tactic_candidates
    
    try:
      random_choice = tactic_candidates[index]
    except:
      self.state.error = "llm error"
      return self

    ###############################
    next_state = self.proof(self.state, random_choice, lean)
    if(next_state.error is not None):
      correct_flag = False
    else:
      correct_flag = True
    ###############################
    next_node = Node(next_state)
    next_node.tac = random_choice
    next_node.flag = correct_flag
    return next_node

  
  
  def __repr__(self):
    return "Node: {}, Q/N: {}/{}, state: {}".format(
        hash(self), self.quality_value, self.visit_times, self.state)


class MCTS:

    def __init__(self, node, policy_model, value_model, args, device):
        self.node = node
        self.policy_model = policy_model
        self.value_model = value_model
        self.args = args
        self.device = device

    def tree_policy(self, node, lean, is_exploration):
      """
      蒙特卡罗树搜索的Selection和Expansion阶段，传入当前需要开始搜索的节点（例如根节点），根据exploration/exploitation算法返回最好的需要expend的节点，注意如果节点是叶子结点直接返回。

      基本策略是先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找权衡过exploration/exploitation的UCB值最大的，如果UCB值相等则随机选。
      """

      # Check if the current node is the leaf node
      while node.is_terminal() == False:
        
        if node.is_all_expand(self.args['TACRIC_NUMBER']):
          # print(node.state.state)
          best_node = self.best_child(node, is_exploration)
          # node = best_node
          
          if(best_node is None):
            print("该节点的子节点的所有策略都无效{}".format(node.state.tacticState))
            node.flag = False
            if(node.parent is not None):
              node = node.parent
            else:
              print("目标无法产生新定理")
              node.llmflag = False
              return node
          else:
            node = best_node
            
        else:
          # Return the new sub node
          sub_node = self.expand(node,lean)
          return sub_node

      # Return the leaf node
      return node


    def expand(self, node, lean):
      """
      输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
      """

      # tried_sub_node_states = [     # 统计node已经展开的所有子节点
      #     sub_node.get_state().state.tacticState for sub_node in node.get_children()
      # ]
      
      # tried_sub_node_tacs = [     # 统计node已经展开的所有子节点
      #     sub_node.get_state().tac for sub_node in node.get_children()
      # ]

      # Check until get the new state which has the different action from others
      # while new_state.state.tacticState in tried_sub_node_states and new_state.tac in tried_sub_node_tacs:  # 判断新状态是否已经被expand，若已经被expand，则重新对node随机采取action，获得新状态
      #   new_state = node.get_next_state_with_random_choice(lean)   # 根据当前node状态随机采取action，获得执行后的新状态
      
      new_node = node.get_next_state_with_random_choice(lean, len(node.children))   # 根据当前node状态随机采取action，获得执行后的新状态
      new_node.depth = node.depth + 1
      #########################
      encodestate = encode_state(node.state.getTacticState(), self.args['feature_size'])
      encodetactic = encode_tactic(new_node.tac, self.args['feature_size'])
      input_policy = encodestate + encodetactic
      input_policy = torch.FloatTensor(np.array(input_policy).astype(np.float64)).to(self.device)
      new_node.prob = float(self.policy_model(input_policy))  # 返回的应该不是值，而是数组？
      #########################
      node.add_child(new_node)
      if(new_node.flag == False):
        new_node.new = False
      else:
        if(new_theorem(new_node)):
          new_node.new = True
        else:
          new_node.new = False
      return new_node


    def best_child(self, node, is_exploration):
      """
      使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
      """

      # TODO: Use the min float value
      best_score = -sys.maxsize
      best_sub_node = None

      for sub_node in node.get_children():
        if(sub_node.flag == False or sub_node.llmflag == False):
          continue
          
        # Ignore exploration for inference
        if is_exploration:
          C = 1 / math.sqrt(2.0)
        else:
          C = 0.01

        # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
        left = sub_node.get_quality_value() / sub_node.get_visit_times()
        right = math.sqrt(node.get_visit_times()) / (sub_node.get_visit_times()+1)
        Puct_score = left + C * sub_node.prob * math.sqrt(right)
        sub_node.puct = Puct_score

        if Puct_score > best_score:
          best_sub_node = sub_node
          best_score = Puct_score
      # print("best:{}".format(best_sub_node.state.state.tacticState))
      return best_sub_node


    def backup(self, node, reward):
      """
      蒙特卡洛树搜索的Backpropagation阶段，输入前面获取需要expend的节点和新执行Action的reward，反馈给expend节点和上游所有节点并更新对应数据。
      """

      # Update util the root node
      while node != None:
        # Update the visit times
        node.visit_times_add_one()

        # Update the quality value
        node.quality_value_add_n(reward)

        # Change the node to the parent node
        node = node.parent


    def run(self, lean):
      node =  self.node
      computation_budget = 1000
      
      # Run as much as possible under the computation budget
      for i in range(computation_budget):

        # 1. Find the best node to expand
        # print("mcts到第{}次，node为：{}".format(i,node.state))
        expand_node = self.tree_policy(node, lean, True)
        
        if(expand_node.llmflag == False):
          return node
        ############################################
        if(not expand_node.flag):
          reward = -1
        elif(expand_node.state.isFinish()):
          reward = 1
        else:
          encodestate = encode_state(expand_node.state.getTacticState(), self.args['feature_size'])
          input_value = torch.FloatTensor(np.array(encodestate).astype(np.float64)).to(self.device)
          reward = float(self.value_model(input_value))

        # 3. Update all passing nodes with reward
        self.backup(expand_node, reward)
        
        if(expand_node.state.isFinish()):
          return node

      return node


    def runmcts(self, lean, time_out):
      count = 0
      node =  self.node
      computation_budget = 1000000
      
      outputs = []
      
      # Run as much as possible under the computation budget
      for i in range(computation_budget):

        # 1. Find the best node to expand
        # print("mcts到第{}次，node为：{}".format(i,node.state))
        expand_node = self.tree_policy(node, lean, True)
        if(expand_node.llmflag == False):
          return outputs
        ############################################
        if(not expand_node.flag):
          reward = -1
        elif(expand_node.new):
          reward = 1
        else:
          encodestate = encode_state(expand_node.state.getTacticState(), self.args['feature_size'])
          input_value = torch.FloatTensor(np.array(encodestate).astype(np.float64)).to(self.device)
          reward = float(self.value_model(input_value))
        

        # 3. Update all passing nodes with reward
        self.backup(expand_node, reward)

        if(expand_node.new): #生成了新定理, 填补证明步骤
          
          new_theorems = generate_theorem(expand_node)
          # print('new_theorem:',new_theorem)

          outputs.append(new_theorems)

          with open('out.json', 'a') as file:
            json.dump(new_theorems, file)
            file.write('\n')                      
          count += 1
          
          if(count>=self.args['max_count']):
            break

      return outputs

