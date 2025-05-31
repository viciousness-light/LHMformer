import torch
from torch import nn

from .layer import *


class LHMformer(nn.Module):

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.num_heads = model_args["num_heads"]
        self.dropout = model_args["attn_drop"]
        self.MLP_drop = model_args["MLP_drop"]
        self.feed_forward_dim = model_args["feed_forward_dim"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.tcn_layers = model_args["tcn_layers"]
        self.model_dim = self.node_dim + self.embed_dim + self.temp_dim_tid + self.temp_dim_diw
        self.long_term_dim = self.node_dim + self.embed_dim
        self.middle_dim = model_args["middle_dim"]
        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.input_len, self.num_nodes, self.node_dim))
            )
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Embedding(self.time_of_day_size, self.temp_dim_tid)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Embedding(self.day_of_week_size, self.temp_dim_diw)
        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)
        self.encoder = DilatedConvEncoder(in_channels=self.input_dim, channels=[self.input_dim] * self.tcn_layers[0] + [24] * self.tcn_layers[1],
                                          kernel_size=3)
        self.long_input_proj = nn.Conv1d(
            in_channels=self.time_of_day_size, out_channels=self.input_len, kernel_size=1, bias=True)

        self.output_proj = nn.Linear(
            self.input_len * self.model_dim, self.input_len
        )

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layer)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layer)
            ]
        )
        self.mlp_net = MLP_Layer(
            input_dim=self.output_len * self.long_term_dim,
            hidden_dims=[self.middle_dim] * 3, 
            output_dim=self.output_len,
            dropout=self.MLP_drop,
            activation=nn.ReLU
        )
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, long_history_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of LHMformer.

        Args:
            long_history_data (torch.Tensor): long-term history data with shape [B, L', N, C]
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        x = history_data[..., range(self.input_dim)]
        x_l = long_history_data[..., range(self.input_dim)]
        batch_size,_,_,_ = x.shape
        x = self.input_proj(x)  

        x_l = x_l.permute(0, 2, 3, 1).reshape(batch_size * self.num_nodes, -1, self.time_of_day_size)
        x_l = self.encoder(x_l)
        x_l = self.long_input_proj(x_l.permute(0,2,1))
        x_l = x_l.reshape(batch_size, self.num_nodes, self.input_len,self.embed_dim).transpose(1,2)

        features = [x]
        features_l = [x_l]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb((t_i_d_data[:, :, :] * self.time_of_day_size).long())
            features.append(time_in_day_emb)

        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb((d_i_w_data[:, :, :] * self.day_of_week_size).long())
            features.append(day_in_week_emb)

        if self.if_spatial:
            node_emb = self.node_emb.expand(
                size=(batch_size, *self.node_emb.shape)
            )
            features.append(node_emb)
            features_l.append(node_emb)

        x = torch.cat(features, dim=-1) 
        x_l = torch.cat(features_l,dim=-1)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)

        out = x.transpose(1, 2)  
        out = out.reshape(
            batch_size, self.num_nodes, self.output_len * self.model_dim)
        out = self.output_proj(out).view(
            batch_size, self.num_nodes, self.output_len, 1)
        out = out.transpose(1, 2) 

        out_l = x_l.transpose(1, 2) 
        out_l = out_l.reshape(
            batch_size, self.num_nodes, self.output_len * self.long_term_dim)
        out_l = self.mlp_net(out_l)
        out_l = out_l.reshape(
            batch_size, self.num_nodes, self.output_len , 1)
        out_l = out_l.transpose(1, 2)
        out = out_l + out
        return out
