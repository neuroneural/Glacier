import torch
import torch.nn as nn

import os
import numpy as np

import math
from .utils import init



class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.15,
        hidden_size: int = 48 * 2,
        num_layers: int = 0,
       
    ):
        super(MLP, self).__init__()
        layers = [

            nn.Linear(input_size, output_size),
        ]


        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # x = x.permute(0,2,1)
        bs, ln, fs = x.shape
        fc_output = self.fc(x.reshape(-1, fs))
        fc_output = fc_output.view(bs, ln, -1)#.mean(1)  # .squeeze(1)
        return fc_output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class combinedModel(nn.Module):
    """Bidirectional LSTM for classifying subjects."""

    def __init__(self, encoder, PT="", exp="UFPT", device_one="cuda", oldpath="",k=10, n_regions=100,device_two="",device_zero="",device_extra=""):

        super().__init__()
        self.encoder = encoder
        # self.graph = graph
        self.samples_per_subject = 1
        self.n_clusters = 4
        # self.auto_encoder = auto_encoder(self.samples_per_subject)
        # self.auto_decoder = auto_decoder(self.samples_per_subject)
        self.w=1
        self.n_regions = n_regions
        self.n_regions_after = n_regions
        self.PT = PT
        self.exp = exp
        self.device_zero = device_zero
        self.device_one = device_one
        self.device_two = device_two
        self.device_extra=device_extra
        self.oldpath=oldpath
        self.time_points=155
        self.division=1 # self.division is used to divide time points into smaller sets. For example for HCP data 1200 timepoints are to many to fit into memory, so we make 3 sets of 400 time points by setting self.division = 3 and self.time_points = 400
        self.n_heads=2
        self.n_heads_temporal=2
        self.embedding_size = 48
        self.attention_embedding = self.embedding_size * self.n_heads
        self.k=10000#k
        self.upscale= .05
        self.upscale2 = 0.5
        self.embedder_output_dim = self.embedding_size * 1
        self.attention_embedding_temporal = self.embedding_size #* self.n_heads_temporal
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.temperature = 2

        self.up_sample = MLP(input_size=self.embedding_size, output_size=self.embedding_size*self.n_heads, num_layers=1).to(device_zero)


        


        self.gta_embed = nn.Sequential(nn.Linear(self.n_regions * self.n_regions, round(self.upscale * self.n_regions * self.n_regions)),
                                        ).to(self.device_two)

        self.gta_norm = nn.Sequential(nn.BatchNorm1d(round(self.upscale * self.n_regions * self.n_regions)), nn.ReLU()).to(self.device_two)

        self.gta_attend = nn.Sequential(nn.Linear(round(self.upscale * self.n_regions * self.n_regions), round(self.upscale2  * self.n_regions * self.n_regions)),
                                         nn.ReLU(),
                                         nn.Linear(round(self.upscale2 * self.n_regions * self.n_regions), 1)).to(self.device_two)


       
        self.gta_dropout = nn.Dropout(0.35)

        


        self.multihead_attn = nn.MultiheadAttention(self.samples_per_subject * self.attention_embedding,
                                                    self.n_heads).to(self.device_two)

        #################################### MHA 2 ###############################################
        self.position_embeddings_rois = nn.Parameter(torch.zeros(1, self.n_regions, self.embedder_output_dim * self.n_heads)).to(self.device_two)
        self.position_embeddings_rois_dropout = nn.Dropout(0.1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.time_points, self.embedder_output_dim)).to(self.device_zero)
        self.embedder = nn.Sequential(
            nn.Linear(1, self.embedder_output_dim),
            nn.Sigmoid(),
            # nn.Linear(self.embedder_output_dim, self.embedder_output_dim),
            # nn.ReLU()
        ).to(self.device_zero)
        # self.pos_encoder = PositionalEncoding(self.embedder_output_dim, 0.1)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedder_output_dim, nhead=self.n_heads_temporal,dim_feedforward=100,dropout=0.1)#.to(self.device_one)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)#.to(self.device_one)

        
        self.relu = torch.nn.ReLU()
        self.HS = torch.nn.Hardsigmoid()
        self.HW = torch.nn.Hardswish()
        self.selu = torch.nn.SELU()
        self.celu = torch.nn.CELU()
        self.tanh = torch.nn.Tanh()
        self.softplus = torch.nn.Softplus(threshold=20)




#        self.init_weight()
#        self.loadModels()

    def init_weight(self, PT="UFPT"):
        # print(self.gain)
        print('init' + PT)
        # return
        if PT == "NPT":
            # for name, param in self.query_layer.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            # for name, param in self.key_layer.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            #     # param = param + torch.abs(torch.min(param))
            # for name, param in self.value_layer.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
                # param = param + torch.abs(torch.min(param))
            for name, param in self.multihead_attn.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param,mode='fan_in')

            # for name, param in self.up_sample.named_parameters():
            #     if 'weight' in name and param.dim() > 1:
            #         # print(param.dim())
            #         nn.init.xavier_normal_(param)

            # for name, param in self.query_layer_temporal.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param, mode='fan_in')
            # for name, param in self.key_layer_temporal.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param, mode='fan_in')
            #     # param = param + torch.abs(torch.min(param))
            # for name, param in self.value_layer_temporal.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param, mode='fan_in')
            #     # param = param + torch.abs(torch.min(param))
            # for name, param in self.multihead_attn_temporal.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param, mode='fan_in')

                # param = param + torch.abs(torch.min(param))

                # param = param + torch.abs(torch.min(param))

           

            for name, param in self.gta_embed.named_parameters():
                # print('name = ',name)
                if 'weight' in name:
                    nn.init.kaiming_normal_(param,mode='fan_in')
                # with torch.no_grad():
                #     param.add_(torch.abs(torch.min(param)))
                    # print(torch.min(param))

            for name, param in self.embedder.named_parameters():
                # print('name = ',name)
                if 'weight' in name and '2' not in name and '7' not in name:
                    nn.init.kaiming_normal_(param,mode='fan_in')

            for name, param in self.encoder_layer.named_parameters():
                # print('name = ',name)
                if 'weight' in name and 'norm' not in name:
                    nn.init.kaiming_normal_(param,mode='fan_in')


            for name, param in self.gta_attend.named_parameters():
                # print(name)

                if 'weight' in name:
                    nn.init.kaiming_normal_(param,mode='fan_in')



        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param,mode='fan_in')

      



   




    def gta_attention(self,x,node_axis=1,outputs='', dimension='time',mode='train'):
        if dimension=='time':

            x_readout = x.mean(node_axis, keepdim=True)
            x_readout = (x*x_readout)
            a = x_readout.shape[0]
            b = x_readout.shape[1]
            x_readout = x_readout.reshape(-1,x_readout.shape[2])
            x_embed = self.gta_norm(self.gta_embed(x_readout))
            x_graphattention = (self.gta_attend(x_embed).squeeze()).reshape(a, b)

            x_graphattention = self.HW(x_graphattention.reshape(a, b)) # You can use hard sigmoid here as well. It might decrease the classification performance, which can be fixed by changing tuning other hyper parameters.


        

            return (x * (x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention


        


    

    

    def multi_head_attention(self, outputs, k, FNC="", FNC2=""):



        # key = self.key_layer(outputs)
        # value = self.value_layer(outputs)
        # query = self.query_layer(outputs)
        # key = key.permute(1,0,2)
        # value = value.permute(1, 0, 2)
        # query = query.permute(1, 0, 2)

        outputs = outputs.permute(1, 0, 2)
        attn_output, attn_output_weights = self.multihead_attn(outputs, outputs, outputs)
        attn_output = attn_output.permute(1,0,2)

        # attn_output_weights = attn_output_weights #+ FNC + FNC2
        return attn_output, attn_output_weights






    def forward(self, input, targets, mode='train', device="cpu",epoch=0, FNC = ""):
        indices = ""


        B = input.shape[0]
        W = input.shape[1]
        R = input.shape[2]
        T = input.shape[3]

        # self.time_points = W
        input = input.reshape(B,self.division,self.time_points,R,T) # self.division is used to divide time points into smaller sets. For example for HCP data 1200 timepoints are to many to fit into memory, so we make 3 sets of 400 time points by setting self.division = 3 and self.time_points = 400
        input = input.permute(1,0,2,3,4)
        (FC_logits), FC,  FC_sum, FC_time_weights = 0., 0., 0., 0.

        for sb in range(self.division):

            sx = input[sb,:,:,:,:]
            B = sx.shape[0]
            W = sx.shape[1]
            R = sx.shape[2]
            T = sx.shape[3]




            inputs = sx.permute(0, 2, 1, 3).contiguous()



            inputs = inputs.reshape(B*R*W,T)

            inputs = (self.embedder(inputs))

            inputs=inputs.to(self.device_one)


           


            inputs = inputs.reshape(B,R,W,-1)

            ########################## transformer encoder - start#################################

            ########################## iterative################################# This is slower but requires less memory
            # coll_list = []
            # temporal_FC = []
            # for sub in inputs:
            #     sub = sub.permute(1, 0, 2).contiguous()
            #     sub, weights = self.transformer_encoder(sub) # to get temporal weights, you need to edit the return statement of the forward functions of transformer encoder and transformer encoder layer to return weight matrix along with emebddings. and edit the function call to multi head attention in the forward function of transformer encoder layer by setting need_weights=True
            #     coll_list.append(sub)
            #     temporal_FC.append(weights)

            # inputs = torch.stack(coll_list)
            # temporal_FC = torch.stack(temporal_FC)
            # # print('temporal FC shape = ', temporal_FC.shape)
            # inputs = inputs.permute(0, 2, 1, 3).contiguous()
            # inputs = inputs.reshape(B * R, W, -1)
            ########################## iterative#################################

            ########################## Non iterative################################# This is faster but can go out of memory

            inputs = inputs.reshape(B * R, W, -1)
            inputs = inputs.permute(1,0,2).contiguous()
            inputs = self.transformer_encoder(inputs) # to get temporal weights, you need to edit the return statement of the forward functions of transformer encoder and transformer encoder layer to return weight matrix along with emebddings. and edit the function call to multi head attention in the forward function of transformer encoder layer by setting need_weights=True
            inputs = inputs.permute(1,0,2).contiguous()

            ########################## Non iterative#################################
            ########################## transformer encoder - end#################################

            # print('4', torch.cuda.memory_reserved() / (1024 ** 3))
            # print('6', torch.cuda.memory_allocated())
            inputs = self.up_sample(inputs)
            inputs = inputs.to(self.device_two)
            inputs = inputs.reshape(B, R, W, -1)
            # print('7', inputs.shape)
            ########################## transformer encoder#################################




            inputs = inputs.permute(2,0,1,3).contiguous()

            inputs = inputs.reshape(W*B,R,-1)

            inputs = self.position_embeddings_rois_dropout(inputs + self.position_embeddings_rois)
            
            _ , attn_weights = self.multi_head_attention(inputs,self.k)

            attn_weights = attn_weights.reshape(W,B,R,R)

            attn_weights = attn_weights.permute(1, 0, 2, 3).contiguous()


            attn_weights = attn_weights.reshape(B, W, -1)


            FC, FC_time_weights = self.gta_attention(attn_weights,dimension='time',mode=mode) #FC_time_weights is the temporal attention weights to create single FC matrix

            FC = FC.squeeze().reshape(B,R,R)
            # FC_sum =  torch.mean(attn_weights,dim=1).squeeze().reshape(B,R,R) # can use the sum of all FC matrices if don't want to use attention based mean


            if sb ==0:
                FC_logits = self.encoder((FC.unsqueeze(1)))

            else:
                FC_logits += self.encoder((FC.unsqueeze(1)))

        kl_loss = 0.

        if mode == 'test':
            return (FC_logits/self.division), kl_loss , FC, "temporal_FC"#, FC_sum, FC_time_weights.squeeze(), attn_weights#, means_logits,selected_indices,ENC_from_means

        return (FC_logits/self.division), kl_loss , FC, "temporal_FC"#, FC_sum, FC_time_weights.squeeze(), 'attn_weights'#, means_logits,selected_indices,ENC_from_means

