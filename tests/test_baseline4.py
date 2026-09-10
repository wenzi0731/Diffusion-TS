from copy import deepcopy
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
from baseline4.data import BASE_WEATHER_COLUMNS, PV_WEATHER_COLUMNS, TARGET_COLUMNS
from baseline4.model import ConditionalDiffusionTS, generate_scenarios
from baseline4.runtime import load_config
from baseline4.sweep import sweep, trial_configs
from baseline4.train import make_dataset, train


@pytest.fixture
def tiny_config(tmp_path):
    torch.set_num_threads(1)
    config = load_config("baseline4/configs/heew.yaml")
    config["model"].update(d_model=16, encoder_layers=1, decoder_layers=1)
    config["diffusion"].update(timesteps=8, sampling_steps=2)
    config["training"].update(epochs=1, batch_size=2, val_every=1)
    config["evaluation"].update(batch_size=2, val_scenarios=2, test_scenarios=3, sample_chunk_size=4)
    config["run"].update(device="cpu", output_root=str(tmp_path / "runs"))
    timestamps = pd.DatetimeIndex([
        ts for year in (2014, 2020, 2021, 2022)
        for ts in pd.date_range(f"{year}-01-01", periods=48, freq="h")
    ])
    cols = dict(Year=timestamps.year, Month=timestamps.month, Day=timestamps.day, Hour=timestamps.hour)
    rng = np.random.default_rng(12)
    energy = pd.DataFrame(cols)
    weather = pd.DataFrame(cols)
    for i, label in enumerate(TARGET_COLUMNS):
        energy[label] = rng.uniform(10, 50, len(timestamps)) * (i + 1)
    for label in BASE_WEATHER_COLUMNS + PV_WEATHER_COLUMNS:
        weather[label] = rng.uniform(0, 1, len(timestamps))
    for kind, frame in (("energy", energy), ("weather", weather)):
        path = tmp_path / f"{kind}.csv"
        frame.to_csv(path, index=False)
        config["data"][f"{kind}_path"] = str(path)
    return config


def test_upstream_output_and_loss_parity_without_condition_embedding(tiny_config):
    model = ConditionalDiffusionTS(tiny_config, 18).eval()
    for parameter in model.condition_embedding.parameters():
        parameter.data.zero_()
    target, noise = torch.randn(2, 24, 4), torch.randn(2, 24, 4)
    t = torch.tensor([2, 5])
    cond, year = torch.randn(2, 24, 18), torch.randn(2, 24, 1)
    torch.testing.assert_close(model.output(target, t), model.denoise(target, t, cond, year))
    expected = Diffusion_TS._train_loss(model, target, t, noise=noise)
    actual = model(target.transpose(1, 2), cond.transpose(1, 2), year.transpose(1, 2), timestep=t, noise=noise)
    torch.testing.assert_close(actual, expected)


def test_condition_gradients_and_repeatable_sampling(tiny_config):
    model = ConditionalDiffusionTS(tiny_config, 18)
    cond, year, target = torch.randn(2, 18, 24), torch.randn(2, 1, 24), torch.randn(2, 4, 24)
    model(target, cond, year).backward()
    assert sum(p.grad.abs().sum().item() for p in model.condition_embedding.parameters()) > 0
    model.eval()
    def sample(c):
        return generate_scenarios(model, c, year, 3, torch.Generator().manual_seed(42), 4)
    first = sample(cond)
    assert first.shape == (2, 3, 4, 24) and torch.isfinite(first).all()
    torch.testing.assert_close(first, sample(cond), rtol=0, atol=0)
    assert not torch.allclose(first, sample(cond + 2))
    assert not torch.allclose(first[:, 0], first[:, 1])


def test_train_only_scaling_and_splits(tiny_config):
    train_data = make_dataset(tiny_config, "train")
    val_data = make_dataset(tiny_config, "val")
    test_data = make_dataset(tiny_config, "test")
    assert len(train_data) == 4 and len(val_data) == len(test_data) == 2
    assert all(d.startswith("2021") for d in val_data.dates)
    assert all(d.startswith("2022") for d in test_data.dates)
    frame = pd.read_csv(tiny_config["data"]["energy_path"])
    train_values = frame.loc[frame.Year <= 2020, TARGET_COLUMNS].to_numpy(np.float32)
    np.testing.assert_allclose(train_data.target_mean, train_values.mean(0), rtol=1e-6)
    np.testing.assert_array_equal(train_data.target_mean, test_data.target_mean)
    np.testing.assert_array_equal(train_data.target_std, val_data.target_std)


def test_six_trials_and_reject_duplicates(tiny_config):
    search = yaml.safe_load(open("baseline4/configs/search_space.yaml"))
    assert len(trial_configs(tiny_config, search)) == 6
    bad = deepcopy(search)
    bad["configurations"].pop()
    with pytest.raises(ValueError, match="six"):
        trial_configs(tiny_config, bad)
    bad = deepcopy(search)
    bad["configurations"][1]["overrides"] = bad["configurations"][0]["overrides"]
    with pytest.raises(ValueError, match="Duplicate"):
        trial_configs(tiny_config, bad)


def test_checkpoint_to_test_outputs(tiny_config, tmp_path, monkeypatch):
    checkpoint = train(tiny_config, "smoke")
    from baseline4.evaluate import main
    out = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(checkpoint),
                                      "--outdir", str(out), "--device", "cpu"])
    main()
    payload = np.load(out / "baseline4_scenarios.npz")
    assert payload["scenarios"].shape == (2, 3, 4, 24)
    metrics = json.loads((out / "global_metrics.json").read_text())
    for label in TARGET_COLUMNS:
        for suffix in ("RMSE_Z", "MAE_Z", "Precision_Z", "Recall_Z", "CR", "IW"):
            assert np.isfinite(metrics[f"{label}_{suffix}"])
    assert metrics["sample_seed"] == 90042
    assert len(list((out / "random_timeseries_50").glob("*.png"))) == 2
    assert (out / "pearson" / "generated_global_pearson.png").exists()
    with pytest.raises(FileExistsError):
        train(tiny_config, "smoke")


def test_sweep_exports_usable_winner(tiny_config, tmp_path):
    search = {"configurations": [
        {"id": f"trial{i}", "overrides": {"model.d_model": 16,
                                         "training.learning_rate": (i + 1) * 1e-5}}
        for i in range(6)
    ]}
    out = tmp_path / "sweep"
    best = sweep(tiny_config, search, out)
    winner = yaml.safe_load((out / "best_config.yaml").read_text())
    assert winner["training"]["learning_rate"] == best["learning_rate"]
    assert len(pd.read_csv(out / "tuning_results.csv")) == 6
    assert sweep(tiny_config, search, out, skip_completed=True) == best
    winner["run"]["seed"] = 123
    assert train(winner).exists()
