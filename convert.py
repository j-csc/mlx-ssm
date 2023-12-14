# Path: torch.py
import torch
import argparse
import numpy as np


def scan_state(state):
    for k, v in state.items():
        print(k, v.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mamba weights to MLX")
    parser.add_argument("--torch_weights", default="pretrained/pytorch_model.bin")
    parser.add_argument("--output_file", default="pretrained/model.npz")
    args = parser.parse_args()

    state = torch.load(args.torch_weights)
    scan_state(state)
    np.savez(
        args.output_file, **{k: v.to(torch.float16).numpy() for k, v in state.items()}
    )
