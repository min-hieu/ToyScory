import abc
import torch
import numpy as np

class SDE(abc.ABC):
    def __init__(self, N: int, T: int):
        super().__init__()
        self.N = N         # number of time step
        self.T = T         # end time
        self.dt = T / N

    @abc.abstractmethod
    def sde_coeff(self, t, x):
        pass

    @abc.abstractmethod
    def marginal_prob(self, t, x):
        pass

    @abc.abstractmethod
    def prior_sampling(self, x):
        pass

    def sde_coeff(self, t, x):
        raise NotImplementedError

    def marginal_prob(self, t, x):
        raise NotImplementedError

    def prior_sampling(self, x):
        return torch.randn(x)

    def dw(self, x, dt=None):
        dt = self.dt if dt is None else dt
        return torch.randn_like(x)*np.sqrt(dt)

    def reverse(self, model):
        N = self.N
        T = self.T
        dt = self.dt
        sde_coeff = self.sde_coeff

        def get_reverse_drift_fn(model_fn):
            def reverse_drift_fn(t, x):
                # TO FILL
                drift, diffusion = sde_coeff(self.T-t, x)
                score = model_fn(self.T-t, x)
                reverse_drift = - drift + score * (diffusion ** 2)
                return reverse_drift
            return reverse_drift_fn

        class RSDE(self.__class__):
            def __init__(self, model):
                self.N = N
                self.T = T
                self.dt = dt
                self.model = model
                self.reverse_drift_fn = get_reverse_drift_fn(model)

            def sde_coeff(self, t, x):
                _, diffusion = sde_coeff(self.T-t, x)
                drift = self.reverse_drift_fn(t, x)
                return drift, diffusion

        return RSDE(model)

class OU(SDE):
    def __init__(self, N=100, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        drift = -0.5 * x
        diffusion = torch.ones(x.shape)
        return drift, diffusion

    def marginal_prob(self, t, x):
        log_mean_coeff = -0.5 * t
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1 - torch.exp((2 * log_mean_coeff)[:,None])) * torch.ones_like(x)
        return mean, std

    def prior_sampling(self, x):
        return torch.randn(shape)

class VESDE(SDE):
    def __init__(self, N=100, T=1, sigma_min=0.01, sigma_max=50):
        super().__init__(N, T)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))

    def sde_coeff(self, t, x):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        log_mean_coeff = torch.tensor(np.log(self.sigma_max) - np.log(self.sigma_min))[:, None]
        diffusion = sigma * torch.sqrt(2 * log_mean_coeff) * torch.ones_like(x)
        return drift, diffusion

    def marginal_prob(self, t, x):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t[:, None]
        mean = x
        return mean, std


class VPSDE(SDE):
    def __init__(self, N=1000, T=1, beta_min=0.1, beta_max=20):
        super().__init__(N, T)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def sde_coeff(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t[:, None])
        return drift, diffusion

    def marginal_prob(self, t, x):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff[:, None])) * torch.ones_like(x)
        return mean, std


class EDM(SDE):
    def __init__(self, N=100, T=1):
        super().__init__(N, T)


class SBFBSDE(SDE):
    def __init__(self, N=1000, T=1,
                 F = lambda xt,t: -0.5 * xt,
                 G = lambda t: torch.ones(x.shape)): # OU coefficients
        super().__init__(N, T)
        self.f = F
        self.g = G

    def sde_coeff(self, t, x):
        raise NotImplementedError

    def marginal_prob(self, t, x):
        raise NotImplementedError

    def prior_sampling(self, shape):
        return torch.randn(shape)

    def diffuse_step(self, t, x, z, dir, f=None, dw=None, dt=None):
        g  = self.g(t)
        f  = self.f(x,t) if f is None else f
        dt = self.dt if dt is None else dt
        dw = self.dw(x,dt) if dw is None else dw

        return x + (dir*f + g*z)*dt + g*dw
