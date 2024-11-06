⁠ from torch.nn import Module
from torch.nn import functional as F

class AdaptiveDBELoss(Module):
    def __init__(self, a1_base=1.5, a2_base=-0.1, epsilon=1e-6):
        super(AdaptiveDBELoss, self).__init__()
        self.a1_base = a1_base
        self.a2_base = a2_base
        self.epsilon = epsilon

    def g(self, d, a1, a2):
        return a1 * d + 0.5 * a2 * (d ** 2)
        
    def forward(self, d_hat, d):
        # Calculating mean and standard deviation
        mu_d = d.mean()
        sigma_d = d.std()

        # Computing adaptive a1 and a2 
        a1 = self.a1_base * (sigma_d / (mu_d + self.epsilon))
        a2 = self.a2_base * (sigma_d / (mu_d + self.epsilon))
        
        # Applying the transformation g(d)
        g_d_hat = self.g(d_hat, a1, a2)
        g_d = self.g(d, a1, a2)

        # Calculate the adaptive DBE Loss
        dbe_loss = 0.5 * F.mse_loss(g_d_hat, g_d, reduction='sum')
        
        return dbe_loss