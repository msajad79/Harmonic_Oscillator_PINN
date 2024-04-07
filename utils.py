
import torch
from matplotlib import pyplot as plt
import pandas as pd

def oscillation(t, wn, zeta, x0=1.0, xdot0=0.0):
    if zeta < 1.0:
        X = torch.sqrt(wn**2.0*x0**2.0 + xdot0**2.0 + 2.0*zeta*wn*x0*xdot0)/(wn*torch.sqrt(1-zeta**2.0))
        phi = torch.atan((xdot0 + zeta*wn*x0)/(x0*wn*torch.sqrt(1-zeta**2.0)))
        
        x = X*torch.exp(-zeta*wn*t)*torch.cos(torch.sqrt(1-zeta**2.0)*wn*t-phi)
    elif zeta > 1.0:
        C1 = (x0*wn*(zeta+torch.sqrt(zeta**2.0-1))+xdot0) / (2.0*wn*torch.sqrt(zeta**2.0-1))
        C2 = (-x0*wn*(zeta-torch.sqrt(zeta**2.0-1))-xdot0) / (2.0*wn*torch.sqrt(zeta**2.0-1))
        
        x = C1*torch.exp((-zeta + torch.sqrt(zeta**2.0-1))*wn*t) + C2*torch.exp((-zeta - torch.sqrt(zeta**2.0-1))*wn*t)
    else:
        C1 = x0
        C2 = xdot0 + x0*wn
        
        x = (C1+C2*t)*torch.exp(-wn*t)
    return x

def plot_pred_exact_oscillation(inp_phy:torch.tensor, out_phy:torch.tensor, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)
    for gp_id in inp_phy[:,5].unique():
        mask_gp = inp_phy[:,5] == gp_id
        gp_inp = inp_phy[mask_gp]
        gp_out = out_phy[mask_gp]
        exact_sol = oscillation(t=gp_inp[:,0], wn=gp_inp[0,2], zeta=gp_inp[0,1], x0=gp_inp[0,3], xdot0=gp_inp[0,4])
        ax.plot(*zip(*sorted(zip(gp_inp[:,0].detach().cpu(), gp_out[:,0].detach().cpu()))), label=f"predict group id : {gp_id}")
        ax.plot(*zip(*sorted(zip(gp_inp[:,0].detach().cpu(), exact_sol.detach().cpu()))), label=f"exact group id : {gp_id}", linewidth=.5)
    #ax.legend()
    return fig, ax

def plot_lbl_data(inp_lbl:torch.tensor, target_lbl:torch.tensor, fig=None, ax=None):    
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)
    for gp_id in inp_lbl[:,5].unique():
        mask_gp = inp_lbl[:,5] == gp_id
        gp_inp = inp_lbl[mask_gp]
        ax.scatter(gp_inp[:,0].detach().cpu(), target_lbl[mask_gp][:,0].detach().cpu(), label=f"lbl data group id : {gp_id}")
    return fig, ax

def calc_error(inp_phy:torch.tensor, out_phy:torch.tensor):
    data = []
    for gp_id in inp_phy[:,5].unique():
        mask_gp = inp_phy[:,5] == gp_id
        gp_inp = inp_phy[mask_gp]
        gp_out = out_phy[mask_gp]
        exact_sol = oscillation(t=gp_inp[:,0], wn=gp_inp[0,2], zeta=gp_inp[0,1], x0=gp_inp[0,3], xdot0=gp_inp[0,4])
        error = torch.nn.functional.mse_loss(exact_sol, gp_out[:,0]) 
        data.append([gp_inp[0,2].item(), gp_inp[0,1].item(), gp_inp[0,3].item(), gp_inp[0,4].item(), gp_id.item(), error.item()])
    df_gp = pd.DataFrame(data, columns=["wn","zeta","x0", "xdot0", "gp_id", "error"])
    return df_gp
if __name__ == "__main__":
    pass