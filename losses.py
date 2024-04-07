import torch
from torch import nn

class OscillationLoss(nn.Module):

    def __init__(self,lambda_lbl:torch.tensor, lambda_IC_x0:torch.tensor, lambda_IC_xdot0:torch.tensor, lambda_GvEq:torch.tensor):
        super(OscillationLoss, self).__init__()
        self.lambda_lbl = lambda_lbl
        self.lambda_IC_x0 = lambda_IC_x0
        self.lambda_IC_xdot0 = lambda_IC_xdot0
        self.lambda_GvEq = lambda_GvEq
        
    
    def forward(self, inp_lbl:torch.tensor, target_lbl:torch.tensor, out_lbl:torch.tensor, inp_phy:torch.tensor, out_phy:torch.tensor):
        # calculate labled data loss
        self.loss_lbl = torch.nn.functional.mse_loss(target_lbl, out_lbl)
        
        # calculate physics data loss
        mask_phy_t0 = inp_phy[:,0] == 0.0
        # calculate grads
        dxdt, _, _, _, _, _ = torch.autograd.grad(out_phy, inp_phy, torch.ones_like(out_phy), create_graph=True)[0].unbind(dim=1)
        d2xdt2, _, _, _, _, _ = torch.autograd.grad(dxdt, inp_phy, torch.ones_like(dxdt), create_graph=True)[0].unbind(dim=1)
        # IC loss
        self.loss_IC_x0 = torch.nn.functional.mse_loss(inp_phy[mask_phy_t0][:,3:4], out_phy[mask_phy_t0])
        self.loss_IC_xdot0 = torch.nn.functional.mse_loss(inp_phy[mask_phy_t0][:,4], dxdt[mask_phy_t0])
        # Governing Eq loss
        self.loss_GvEq = torch.mean(( d2xdt2 + 2.0*inp_phy[:,1]*inp_phy[:,2]*dxdt + inp_phy[:,2]**2.0*out_phy[:,0] )**2.0)

        # TODO:balnce loss

        # store important variable
        
        
        return (
            self.loss_lbl * self.lambda_lbl +
            self.loss_IC_x0 * self.lambda_IC_x0 + 
            self.loss_IC_xdot0 * self.lambda_IC_xdot0 + 
            self.loss_GvEq * self.lambda_GvEq
        )
    
