import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import random
import matplotlib.pyplot as plt
import time
import numpy as np
import plotly.express as px

import transformer_lens.utils as utils

DEVICE = torch.device('cpu')









def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()


def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    ).show()


import einops


from torch import nn
import os, sys
import torch.nn.functional as F


from transformer_lens import HookedTransformer, utils
import torch
from datasets import load_dataset
from typing import Dict
from tqdm.notebook import tqdm
import plotly.express as px
import json

device = torch.device("cpu")

torch.set_grad_enabled(False)

def get_batch_tokens(dataset_iter, batch_size, model, ctx_len):
    tokens = []
    total_tokens = 0
    while total_tokens < batch_size*ctx_len:
        try:
            # Retrieve next item from iterator
            row = next(dataset_iter)["text"]
        except StopIteration:
            # Break the loop if dataset ends
            break

        # Tokenize the text with a check for max_length
        cur_toks = model.to_tokens(row)
        tokens.append(cur_toks)

        total_tokens += cur_toks.numel()

    # Check if any tokens were collected
    if not tokens:
        return None

    # Depending on your model's tokenization, you might need to pad the tokens here

    # Flatten the list of tokens
    flat_tokens = torch.cat(tokens, dim=-1).flatten()
    flat_tokens = flat_tokens[:batch_size * ctx_len]
    reshaped_tokens = einops.rearrange(
        flat_tokens,
        "(batch seq_len) -> batch seq_len",
        batch=batch_size,
    )
    reshaped_tokens[:, 0] = model.tokenizer.bos_token_id
    return reshaped_tokens

def shuffle_data(all_tokens):
    print("Shuffled data")
    return all_tokens[torch.randperm(all_tokens.shape[0])]

dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
dataset_iter = iter(dataset)
ctx_len = 128
all_tokens_batches = int(1e7) // ctx_len




