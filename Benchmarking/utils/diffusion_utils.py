import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from glide_text2im.gaussian_diffusion import get_named_beta_schedule



class attention_encoder(nn.Module):
    def __init__(self, in_size, dim=768):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_size, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        A = self.attention(x)
        A = A.masked_fill((x == 0).all(dim=2).reshape(A.shape), -9e15)
        A = F.softmax(A, dim=1)
        M = torch.einsum('b k d, b k o -> b o', A, x)
        return M


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / half)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ConditionalDiffusionPatchModel(nn.Module):
    def __init__(self, cond_dim, tgt_dim, time_emb_dim=128, hidden_dim=256):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        self.to_time_hidden = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, time_emb_dim),
        )
        self.to_cond_hidden = nn.Sequential(
            attention_encoder(cond_dim, cond_dim),
            nn.Linear(cond_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(tgt_dim + time_emb_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tgt_dim),
        )

    def forward(self, y_noisy, cond_x, t):
        t_emb = self.time_emb(t)
        t_hidden = self.to_time_hidden(t_emb)
        c_hidden = self.to_cond_hidden(cond_x)
        inp = torch.cat([y_noisy, t_hidden, c_hidden], dim=-1)
        return self.net(inp)


class PatchGaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, num_steps=1000, beta_schedule="linear"):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        betas = get_named_beta_schedule(beta_schedule, num_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def q_sample(self, y_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(y_start)
        a = self.sqrt_alphas_cumprod[t].view(-1, 1)
        am1 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return a * y_start + am1 * noise, noise

    def p_losses(self, y_start, cond_x, t):
        y_noisy, noise = self.q_sample(y_start, t)
        pred_noise = self.model(y_noisy, cond_x, t)
        return F.mse_loss(pred_noise, noise)

    def forward(self, y_start, cond_x):
        B = y_start.size(0)
        t = torch.randint(0, self.num_steps, (B,), device=y_start.device)
        return self.p_losses(y_start, cond_x, t)

    @torch.no_grad()
    def sample(self, cond_x):
        B, K, _ = cond_x.shape
        device = cond_x.device
        y = torch.randn((B, K, self.model.net[-1].out_features), device=device)
        for i in reversed(range(self.num_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            pred_noise = self.model(y, cond_x, t)
            beta = self.betas[t].view(-1, 1, 1)
            alpha = 1 - beta
            y = (1 / torch.sqrt(alpha)) * (y - beta / torch.sqrt(1 - self.alphas_cumprod[t]).view(-1, 1, 1) * pred_noise)
            if i > 0:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta).view(-1, 1, 1) * noise
        return y
