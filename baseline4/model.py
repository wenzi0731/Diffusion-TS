"""Reuse upstream decomposition, x0 objective, Fourier loss and diffusion schedule.

Only the external condition embedding and explicit conditional sampler are new.
The upstream files under Models/ are not modified by this adaptation.
"""
from __future__ import annotations

import torch
from torch import nn

from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
from Models.interpretable_diffusion.model_utils import Conv_MLP, extract


class ConditionalDiffusionTS(Diffusion_TS):
    def __init__(self, config: dict, condition_channels: int):
        model = config["model"]
        diffusion = config["diffusion"]
        seq_len = int(config["data"]["seq_len"])
        if seq_len != 24:
            raise ValueError("The HEEW benchmark requires 24-hour trajectories.")
        timesteps = int(diffusion["timesteps"])
        sampling_steps = int(diffusion["sampling_steps"])
        if not 1 <= sampling_steps <= timesteps:
            raise ValueError("Require 1 <= sampling_steps <= timesteps.")
        super().__init__(
            seq_length=seq_len, feature_size=4,
            n_layer_enc=int(model["encoder_layers"]),
            n_layer_dec=int(model["decoder_layers"]),
            d_model=int(model["d_model"]), n_heads=int(model["heads"]),
            mlp_hidden_times=int(model["mlp_hidden_times"]),
            timesteps=timesteps, sampling_timesteps=sampling_steps,
            beta_schedule="cosine", loss_type="l1", eta=0.0,
            use_ff=True, reg_weight=model.get("fourier_weight"),
            attn_pd=float(model["dropout"]), resid_pd=float(model["dropout"]),
        )
        self.condition_channels = condition_channels
        self.condition_embedding = Conv_MLP(
            condition_channels + 1, int(model["d_model"]),
            resid_pdrop=float(model["dropout"]),
        )

    def denoise(self, noisy, timestep, conditions, pv_year):
        """All sequence tensors here use [batch, hour, channel]."""
        if noisy.shape[1:] != (24, 4):
            raise ValueError("noisy must have shape [batch,24,4]")
        if conditions.shape != (noisy.shape[0], 24, self.condition_channels):
            raise ValueError("Unexpected condition shape")
        if pv_year.shape != (noisy.shape[0], 24, 1):
            raise ValueError("Unexpected PV year feature shape")
        backbone = self.model
        # Clean exogenous conditions remain fixed at every reverse step.
        emb = backbone.emb(noisy) + self.condition_embedding(
            torch.cat([conditions, pv_year], dim=-1)
        )
        encoded = backbone.encoder(backbone.pos_enc(emb), timestep)
        output, mean, trend, season = backbone.decoder(
            backbone.pos_dec(emb), timestep, encoded
        )
        residual = backbone.inverse(output)
        residual_mean = residual.mean(dim=1, keepdim=True)
        seasonal = backbone.combine_s(season.transpose(1, 2)).transpose(1, 2)
        seasonal = seasonal + residual - residual_mean
        trend = backbone.combine_m(mean) + residual_mean + trend
        return trend + seasonal

    def forward(self, target, conditions, pv_year, *, timestep=None, noise=None):
        """Public training interface uses [batch, channel, hour]."""
        target = target.transpose(1, 2)
        conditions, pv_year = conditions.transpose(1, 2), pv_year.transpose(1, 2)
        if timestep is None:
            timestep = torch.randint(self.num_timesteps, (len(target),), device=target.device)
        noisy = self.q_sample(target, timestep, noise=noise)
        prediction = self.denoise(noisy, timestep, conditions, pv_year)
        reconstruction = self.loss_fn(prediction, target, reduction="none")
        # Match upstream: full temporal FFT, real + imaginary L1, norm='forward'.
        pred_fft = torch.fft.fft(prediction, dim=1, norm="forward")
        true_fft = torch.fft.fft(target, dim=1, norm="forward")
        fourier = (pred_fft.real - true_fft.real).abs() + (pred_fft.imag - true_fft.imag).abs()
        per_sample = (reconstruction + self.ff_weight * fourier).mean(dim=(1, 2))
        return (per_sample * extract(self.loss_weight, timestep, per_sample.shape)).mean()

    @torch.no_grad()
    def sample_conditions(self, conditions, pv_year, *, rng, sampling_steps=None):
        """DDIM eta=0; randomness is in initial noise, with no z-score clipping."""
        steps = self.sampling_timesteps if sampling_steps is None else int(sampling_steps)
        if not 1 <= steps <= self.num_timesteps:
            raise ValueError("Invalid sampling_steps")
        condition_seq = conditions.transpose(1, 2)
        year_seq = pv_year.transpose(1, 2)
        sample = torch.randn(
            (len(conditions), self.seq_length, 4), device=conditions.device,
            dtype=conditions.dtype, generator=rng,
        )
        times = torch.linspace(-1, self.num_timesteps - 1, steps + 1).int().tolist()[::-1]
        for step, next_step in zip(times[:-1], times[1:]):
            timestep = torch.full((len(sample),), step, device=sample.device, dtype=torch.long)
            x0 = self.denoise(sample, timestep, condition_seq, year_seq)
            if next_step < 0:
                sample = x0
            else:
                noise = self.predict_noise_from_start(sample, timestep, x0)
                alpha_next = self.alphas_cumprod[next_step]
                sample = alpha_next.sqrt() * x0 + (1 - alpha_next).sqrt() * noise
        return sample.transpose(1, 2)


@torch.no_grad()
def generate_scenarios(model, conditions, pv_year, count, rng, chunk_size=128):
    """Return [day, scenario, channel, hour]; bound GPU sampling memory."""
    if count <= 0 or chunk_size <= 0:
        raise ValueError("Scenario count and chunk size must be positive")
    batch = len(conditions)
    cond = conditions.repeat_interleave(count, dim=0)
    year = pv_year.repeat_interleave(count, dim=0)
    result = []
    for start in range(0, len(cond), chunk_size):
        result.append(model.sample_conditions(
            cond[start:start + chunk_size], year[start:start + chunk_size], rng=rng
        ))
    return torch.cat(result).reshape(batch, count, 4, 24)
