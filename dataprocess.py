import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from utils import oscillation

class LabeledDataset(Dataset):
    def __init__(self,  t_s=torch.linspace(0,0.5,5), zeta_s=torch.tensor([10.0]), wn_s=torch.tensor([10.0]), x0_s=torch.tensor([1.0]), xdot0_s=torch.tensor([0.0]), device="cpu"):
        super().__init__()

        one_t = torch.ones((t_s.shape[0], 1))
        data = torch.tensor([])
        group_id = 0
        for x0 in x0_s:
            for xdot0 in xdot0_s:
                for wn in wn_s:
                    for zeta in zeta_s:
                        group_id += 1
                        x = oscillation(t_s, wn, zeta=zeta, x0=x0, xdot0=xdot0)
                        data = torch.cat([
                            data,
                            torch.cat([
                                x.view(-1,1),
                                t_s.view(-1,1), 
                                torch.full_like(one_t, zeta), 
                                torch.full_like(one_t, wn), 
                                torch.full_like(one_t, x0), 
                                torch.full_like(one_t, xdot0),
                                torch.full_like(one_t, group_id)
                            ], dim=1)
                        ], dim=0)
        self.data = data.to(device)
        self.df = pd.DataFrame(data, columns=["x", "t", "zeta", "wn", "x0", "xdot0", "group_id"])
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx,1:], self.data[idx,0:1]
    
class PhysicsDataset(Dataset):
    def __init__(self, t_s=torch.linspace(0,0.5,5), zeta_s=torch.tensor([10.0]), wn_s=torch.tensor([10.0]), x0_s=torch.tensor([1.0]), xdot0_s=torch.tensor([0.0]), device="cpu"):
        super().__init__()

        # generate data
        one_t = torch.ones((t_s.shape[0], 1))
        data = torch.tensor([])
        group_id = 0
        for x0 in x0_s:
            for xdot0 in xdot0_s:
                for wn in wn_s:
                    for zeta in zeta_s:
                        group_id += 1
                        data = torch.cat([
                            data,
                            torch.cat([
                                t_s.view(-1,1), 
                                torch.full_like(one_t, zeta), 
                                torch.full_like(one_t, wn), 
                                torch.full_like(one_t, x0), 
                                torch.full_like(one_t, xdot0),
                                torch.full_like(one_t, group_id)
                            ], dim=1)
                        ], dim=0)
        self.data = data.requires_grad_(True).to(device)
        self.df = pd.DataFrame(data.detach().cpu(), columns=["t", "zeta", "wn", "x0", "xdot0", "group_id"])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx,:]
    

if __name__ == "__main__":
    ds = LabeledDataset(zeta_s=torch.tensor([0.0,2,5]))
    