# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:36:03 2019

@author: hasee
"""
import os
import pickle
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict

path='D:\\NLP\\'
if not os.path.exists(path):
    os.makedirs(path)
with open (path+'word_embed.pkl','rb') as f:
   word_embed= pickle.load(f)
with open (path+'arr.pkl','rb') as f:
   arr= pickle.load(f)
with open(path+'word2index.pkl','rb') as f:
    word2index=pickle.load(f)
def mask_padding(arr):
    mask=np.zeros((arr.shape))
    mask[arr>0]=1
    return torch.from_numpy(mask).float()
mask=mask_padding(arr)
print(mask[0].size())
squelength=len(arr[0])

class cnn_autoencoder(nn.Module):
    def __init__(self,embedding_pretrain=word_embed,pre_train=True,squelength=squelength):
        super(cnn_autoencoder,self).__init__()
        self.word_num,self.emb_size=embedding_pretrain.shape
        emb=embedding_pretrain
        self.pre_train=pre_train
        
        self.embeding=nn.Embedding(self.word_num,self.emb_size)
        for p in self.embeding.parameters():
            p.requres_grad=False
        self.squelength=squelength
        if pre_train:
            self.embeding.weight.data.copy_(torch.from_numpy(emb).float())
        else:
            self.embinit()
       
        self.con1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(self.emb_size,self.emb_size//3, 1)),
            ('norm1', nn.BatchNorm1d(self.emb_size//3)),
            ('relu1', nn.LeakyReLU(inplace=True)),
            
            ]))
        
        self.con2=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(self.emb_size//3,self.emb_size//6, 1)),
            ('norm1', nn.BatchNorm1d(self.emb_size//6)),
            ('relu1', nn.LeakyReLU(inplace=True)),]))
        self.con3= nn.Sequential(OrderedDict([   
            ('conv1',nn.Conv1d(self.emb_size//6,self.emb_size//12,3,2,1)),
            ('norm1', nn.BatchNorm1d(self.emb_size//12)),
            ('relu1', nn.LeakyReLU(inplace=True)),]))
            
        self.con4= nn.Sequential(OrderedDict([  ('conv1',nn.Conv1d(self.emb_size//12,10,1)),
            ('norm1', nn.BatchNorm1d(10)),('relu1', nn.LeakyReLU(inplace=True)),

            
            ]))
        self.con5=nn.Sequential(OrderedDict([
            ('conv1', nn.ConvTranspose1d(10,self.emb_size//12, 3,2)),
            ('norm1', nn.BatchNorm1d(self.emb_size//12)),
            ('relu1', nn.LeakyReLU(inplace=True)),]))  
        self.con6=nn.Sequential(OrderedDict([    
            ('conv1',nn.ConvTranspose1d(self.emb_size//12,self.emb_size//6,2,2)),
            ('norm1', nn.BatchNorm1d(self.emb_size//6)),
            ('relu1', nn.LeakyReLU(inplace=True)),]))   
        
        self.con7=nn.Sequential(OrderedDict([  
            ('conv1',nn.ConvTranspose1d(self.emb_size//6,self.emb_size//3,1)),
            ('norm1', nn.BatchNorm1d(self.emb_size//3)),
             ('relu1', nn.LeakyReLU(inplace=True))]))  
        
        self.con8=nn.Sequential(OrderedDict([   
            ('conv1', nn.ConvTranspose1d(self.emb_size//3,self.emb_size, 1)),
            ('norm1', nn.BatchNorm1d(self.emb_size)),
            ('relu1', nn.LeakyReLU(inplace=True)),
            ]))        


    def forward(self,x,mask):
        x=self.embeding(x)
        
        position_code=self.get_sinusoid_encoding_table(self.squelength,self.emb_size,padding_idx=None)
        
        input_tensor=x+position_code.view(1,self.squelength,self.emb_size).expand_as(x)
        
        
        input_tensor=input_tensor*mask
        input1=self.con1(input_tensor.permute(0,2,1))
        input2=self.con2(input1)
        input3=self.con3(input2)
        input4=self.con4(input3)
        input4=self.kmax_pooling(input4,2,5)
        input5=self.con5(input4)
        input6=self.con6(input5+input3)
        input7=self.con7(input6+input2)
        input8=self.con8(input7+input1).permute(0,2,1)
        return input8,input4
        
        
        
        






    def embinit(self):
        self.embeding.weight.data.copy_(torch.from_numpy(np.random.uniform(-3,3,(self.word_num,self.emb_size))).float())
    def kmax_pooling(self,x, dim, k):
        index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
        return x.gather(dim, index) 
               
        
        
        
    


    def get_sinusoid_encoding_table(self,n_position, d_hid, padding_idx=None):
        ''' Sinusoid position encoding table '''
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
    
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    
        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.
    
        return torch.FloatTensor(sinusoid_table)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
        
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    
    if padding_idx is not None:
            # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    
    return torch.FloatTensor(sinusoid_table)

trainy=np.zeros((len(arr),len(arr[0]),300))

for i in range(len(arr)):
    for j in range(len(arr[0])):
       trainy[i,j]= word_embed[arr[i,j]]
position=get_sinusoid_encoding_table(len(arr[0]),300)
trainy=torch.from_numpy(trainy).float()
mask=mask.view(mask.size()[0],mask.size()[1],1).expand_as(trainy)
train_y=(trainy+position.view(1,len(arr[0]),300).expand_as(trainy))*mask
model=cnn_autoencoder(word_embed,True,squelength)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)        

trainx=torch.from_numpy(arr).long()


loss=torch.nn.MSELoss()
totalloss=0
epoach=1
patients=7
bestloss=1000
losseformer=10
count=0
for j in range(epoach):
    for i in range(len(trainx)):
        input1,input2=model(trainx[i].unsqueeze(0),mask[i].unsqueeze(0))
        losses=loss(input1.squeeze(0),trainy[i])
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if losses.data.item()<bestloss:
            bestloss=losses.data.item()
        totalloss+=losses.data.item()
        if i %500==0:
            print('epoach',j,'the sentence:',i,'of:',len(trainx),'loss',losses.data.item(),'bestloss:',bestloss,'totalloss:',totalloss)
    if losses.data.item()-losseformer>0.001:
        count+=1
        if count>7:
            break
            
        losseformer=losses.data.item()

torch.save(model.state_dict(),'the train epoach of{}.pth'.format(j))
























