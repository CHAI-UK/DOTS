import torch
from torch import nn

class DiffMLP(nn.Module):
    def __init__(self, n_nodes: int, depth_mul : int = 3) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        big_layer = 1024#max(1024, 5*self.n_nodes)
        small_layer = max(128, big_layer)
        mid_layers = []
        for i in range(depth_mul):
            mid_layers.append(nn.LayerNorm([big_layer]))
            mid_layers.append(nn.Linear(big_layer,big_layer))
            mid_layers.append(nn.LeakyReLU())
            
        self.main_block = nn.Sequential(
            nn.Linear(self.n_nodes + 1, small_layer, bias= False), nn.LeakyReLU(),
            nn.LayerNorm([small_layer]), nn.Dropout(0.2), nn.Linear(small_layer, big_layer), nn.LeakyReLU(), 
            *mid_layers,
            nn.Linear(big_layer,small_layer), nn.LeakyReLU(),
            nn.Linear(small_layer,self.n_nodes),
        )


    def forward(self, X, t):
        X_t = torch.cat([X,t.unsqueeze(1)],axis = 1)
        return self.main_block(X_t)
    

class DiffMLP_base(nn.Module):
    def __init__(self, n_nodes: int, depth_mul : int = 5, big_layer : int = 512) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        mid_layers = []
        for i in range(depth_mul):
            mid_layers.append(residual_MLP(big_layer))
            
        self.main_block = nn.Sequential(
            nn.Linear(self.n_nodes, big_layer), nn.Softplus(),
            *mid_layers,
            nn.Linear(big_layer,self.n_nodes),
        )

    def forward(self, X):
        return self.main_block(X)
    
class residual_MLP(nn.Module):
    def __init__(self, hidden_units: int) -> None:
        super().__init__()
            
        self.main_block = nn.Sequential(
            nn.LayerNorm([hidden_units]), 
            nn.Dropout(0.1), 
            nn.Linear(hidden_units, hidden_units), 
            nn.Softplus(), 
        )

    def forward(self, X):
        return self.main_block(X)