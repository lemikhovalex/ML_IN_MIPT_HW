
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        # <YOUR CODE HERE> 
           
        self.title_conv = nn.Conv1d(in_channels=hid_size, out_channels=hid_size, kernel_size=2)
        self.title_pool = nn.AdaptiveAvgPool1d(output_size=1) 
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        
        # <YOUR CODE HERE> emb->conv->pool
        self.full_conv = nn.Conv1d(in_channels=hid_size, out_channels=hid_size, kernel_size=2)
        self.full_pool = nn.AdaptiveAvgPool1d(output_size=1)
        
        #self.category_out =  Flatten()  # <YOUR CODE HERE>
        

        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)
        self.fin_ac = nn.ReLU()
        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.full_pool(title_beg)# <YOUR CODE HERE>
        
        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.full_pool(full_beg)# <YOUR CODE HERE>        
        
        category = input3 #self.category_out(input3)# <YOUR CODE HERE>        
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        out = self.inter_dense(concatenated)# <YOUR CODE HERE>
        out = self.fin_ac(out)
        out = self.final_dense(out)
        
        return out