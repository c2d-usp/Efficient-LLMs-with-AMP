import argparse
import gc
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from amp_criterion import (measure_amp_heads_importance,
                           measure_amp_mlps_importance)


ARCHITECTURE_SETTINGS = {
    "llama": {
        "head_dim": 128,
        "attn_proj": "o_proj",
        "mlp": {
            "proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj"
        }
    }
}


def select_indices(importance_array, num_to_select, criterion):
    """
    Selects indices based on the chosen criterion.

    - "standard": selects indices with the smallest values.
    - "reversed": selects indices with the largest values.
    - "random": selects indices randomly.
    """
    if criterion == "standard":
        indices = np.argsort(importance_array)
        return indices[:num_to_select]
    elif criterion == "reversed":
        indices = np.argsort(importance_array)[::-1]
        return indices[:num_to_select]
    elif criterion == "random":
        indices = np.arange(len(importance_array))
        return np.random.choice(indices, size=num_to_select, replace=False)
    else:
        raise ValueError("Invalid pruning criterion specified.")


def get_num_heads(layer, arch):
    """
    Returns the number of attention heads in the given layer.
    For llama, the layer stores its num_heads.
    """
    if arch == "llama":
        return layer.self_attn.num_heads
    else:
        raise ValueError("Unsupported architecture: " + arch)


def get_intermediate_size(layer, arch):
    """
    Returns the intermediate (MLP) size for the given layer.
    For llama, uses mlp.gate_proj.out_features.
    """
    if arch == "llama":
        return layer.mlp.gate_proj.out_features
    else:
        raise ValueError("Unsupported architecture: " + arch)


def print_model_parameters(model, message):
    """
    Prints and returns the total number of parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{message}: {total_params} parameters")
    return total_params


def compute_final_pruning_ratio(model, iteration):
    """
    Computes the pruning ratio based on the model and iteration.
    """
    total_params = sum(p.numel() for p in model.parameters())
    params_by_layer = sum(p.numel() for p in model.model.layers[0].parameters())
    params_to_prune = iteration * params_by_layer
    computed_ratio = params_to_prune / total_params
    print(f"Computed pruning ratio: {computed_ratio:.4f}")
    return computed_ratio


# Pruning Functions for Attention & MLP
def prune_attention_heads(attn_layer, heads_to_prune, arch):
    """
    Prunes specified attention heads from the attention layer.
    The implementation differs by architecture.

    For llama:
      - Uses the layer's num_heads and head_dim.
      - Prunes weights in q_proj, k_proj, v_proj and columns in o_proj.
      - Bias updates (for q_proj, k_proj, v_proj) are omitted per your original code.
    """
    if arch == "llama":
        num_heads = attn_layer.num_heads
        head_dim = attn_layer.head_dim
        heads_to_keep = sorted(set(range(num_heads)) - set(heads_to_prune))
        idxs_to_keep = []
        for head in heads_to_keep:
            idxs = list(range(head * head_dim, (head + 1) * head_dim))
            idxs_to_keep.extend(idxs)
        idxs_to_keep = sorted(idxs_to_keep)
        attn_layer.q_proj.weight = torch.nn.Parameter(attn_layer.q_proj.weight.data[idxs_to_keep, :])
        attn_layer.k_proj.weight = torch.nn.Parameter(attn_layer.k_proj.weight.data[idxs_to_keep, :])
        attn_layer.v_proj.weight = torch.nn.Parameter(attn_layer.v_proj.weight.data[idxs_to_keep, :])
        attn_layer.o_proj.weight = torch.nn.Parameter(attn_layer.o_proj.weight.data[:, idxs_to_keep])
        new_num_heads = len(heads_to_keep)
        attn_layer.num_heads = new_num_heads
        attn_layer.hidden_size = new_num_heads * head_dim
        attn_layer.q_proj.out_features = attn_layer.hidden_size
        attn_layer.k_proj.out_features = attn_layer.hidden_size
        attn_layer.v_proj.out_features = attn_layer.hidden_size
        attn_layer.o_proj.in_features = attn_layer.hidden_size
    else:
        raise ValueError("Unsupported architecture in prune_attention_heads")


def prune_mlp_units(mlp_layer, units_to_prune, arch):
    """
    Prunes specified units in the MLP layer.

    For llama:
      - Uses mlp.gate_proj, mlp.up_proj, and mlp.down_proj.
    """
    if arch == "llama":
        intermediate_size = mlp_layer.gate_proj.out_features
        keep_idxs = sorted(set(range(intermediate_size)) - set(units_to_prune))
        mlp_layer.gate_proj.out_features = len(keep_idxs)
        mlp_layer.gate_proj.weight = torch.nn.Parameter(mlp_layer.gate_proj.weight.data[keep_idxs, :])
        mlp_layer.up_proj.out_features = len(keep_idxs)
        mlp_layer.up_proj.weight = torch.nn.Parameter(mlp_layer.up_proj.weight.data[keep_idxs, :])
        mlp_layer.down_proj.in_features = len(keep_idxs)
        mlp_layer.down_proj.weight = torch.nn.Parameter(mlp_layer.down_proj.weight.data[:, keep_idxs])
    else:
        raise ValueError("Unsupported architecture in prune_mlp_units")


# Main Pruning Routine Functions

def prune_model(model, tokenizer, final_pruning_ratio, arch, criterion="standard",  max_prompts=50):
    """
    Prunes the model's attention heads and MLP units.

    Steps:
      1. Calculate total parameters in attention and MLP layers.
      2. Compute a pruning ratio.
      3. Use AMP measurement functions (for heads and MLPs) to get importance scores.
      4. For each layer, determine how many heads and MLP units to prune, select indices
         based on the chosen criterion, and call the respective pruning functions.
    """
    # Calculate total parameters in attention and MLP layers.
    total_attention_params = 0
    total_mlp_params = 0
    for layer in model.model.layers:
        total_attention_params += sum(p.numel() for p in layer.self_attn.parameters())
        total_mlp_params += sum(p.numel() for p in layer.mlp.parameters())
    total_params = total_attention_params + total_mlp_params
    actual_total_params = sum(p.numel() for p in model.parameters())

    # Compute the ratio for pruning.
    pruning_ratio = (actual_total_params * final_pruning_ratio) / total_params

    # Get importance scores using AMP functions.
    amp_head_importances = measure_amp_heads_importance(
        model,
        tokenizer,
        arch,
        dataset_name="yahma/alpaca-cleaned",
        split="train",
        max_prompts=max_prompts,
        random_subset=False
    )

    amp_mlps_importance = measure_amp_mlps_importance(
        model,
        tokenizer,
        arch,
        dataset_name="yahma/alpaca-cleaned",
        split="train",
        max_prompts=max_prompts,
        random_subset=False
    )

    # Iterate over each transformer layer.
    for layer_num, layer in enumerate(model.model.layers):
        # --- Attention Heads Pruning ---
        num_heads = get_num_heads(layer, arch)
        num_prune_heads = round(pruning_ratio * num_heads)
        if num_prune_heads >= num_heads:
            num_prune_heads = num_heads - 1

        head_importance_layer = amp_head_importances[layer_num]
        heads_to_prune = select_indices(head_importance_layer, num_prune_heads, criterion)
        prune_attention_heads(layer.self_attn, heads_to_prune, arch)

        # --- MLP Units Pruning ---
        intermediate_size = get_intermediate_size(layer, arch)
        num_prune_mlp_units = round(
            intermediate_size * ((num_heads / intermediate_size) * (pruning_ratio - (num_prune_heads / num_heads)) + pruning_ratio)
        )
        if num_prune_mlp_units >= intermediate_size:
            num_prune_mlp_units = intermediate_size - 1

        mlp_importance_layer = amp_mlps_importance[layer_num]
        mlp_units_to_prune = select_indices(mlp_importance_layer, num_prune_mlp_units, criterion)
        prune_mlp_units(layer.mlp, mlp_units_to_prune, arch)

    return model


def prune_groups(args, base_dir, arch, criterion):
    """
    Main function to load the model, perform pruning, update configuration, and save the pruned model.
    """
    iteration = args.iteration
    # Load model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print_model_parameters(model, "Total parameters before pruning")

    # Compute and override the final pruning ratio.
    final_pruning_ratio = compute_final_pruning_ratio(model, iteration)
    print("Iteration pruning ratio:", final_pruning_ratio)

    model = prune_model(model, tokenizer, final_pruning_ratio, arch, criterion)
    print_model_parameters(model, "Total parameters after pruning")

    # Update model configuration based on architecture.
    if arch == "llama":
        new_num_heads = model.model.layers[0].self_attn.num_heads
        model.config.num_attention_heads = new_num_heads
        model.config.num_key_value_heads = new_num_heads
        new_intermediate_size = get_intermediate_size(model.model.layers[0], arch)
        model.config.intermediate_size = new_intermediate_size

    # Save the pruned model.
    prune_model_dir = os.path.join(base_dir, f'prune_iter_{iteration}')
    os.makedirs(prune_model_dir, exist_ok=True)
    model.save_pretrained(prune_model_dir)
    tokenizer.save_pretrained(prune_model_dir)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning for LLaMA Models using AMP')
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--save_dir', type=str, required=True, help='Save directory')
    parser.add_argument('--arch', type=str, choices=["llama"], required=True, help='Model architecture')
    parser.add_argument('--criterion', type=str, choices=["standard", "reversed", "random"],
                        default="standard", help='Pruning criterion to use')
    parser.add_argument('--device', type=str, default="cuda", help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument('--iteration', type=int, default=1)
    args = parser.parse_args()
    base_dir = args.save_dir
    prune_groups(args, base_dir, args.arch, args.criterion)
