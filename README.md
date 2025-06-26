# AMP: Attention Heads and MLP Pruning

This repository contains code for pruning large language models (LLMs) using the **A**ttention Heads and **M**LP **P**runing (AMP) method. The implementation currently supports LLaMA models, with a structure that allows for easy extension to other architectures in the future. Models are available at: [Hugging Face collection](https://huggingface.co/collections/c2d-usp/efficient-large-language-models-67a36f099af36b0c122877b0).

Demonstration of AMP (Comparison of Original vs. Pruned LLaMA-2 7B):



https://github.com/user-attachments/assets/b953ea73-5ba7-4c79-b161-bf4c4859174a




## Overview

AMP is a pruning technique that identifies and removes less important components (attention heads and MLP neurons) from transformer-based language models. The importance of these components is determined by measuring their activation patterns during inference on a set of prompts.

The pipeline consists of three main steps:
1. **Pruning**: Identifying and removing less important attention heads and MLP neurons
2. **Fine-tuning**: Training the pruned model using LoRA to recover performance
3. **Evaluation**: Assessing the pruned and fine-tuned model on standard benchmarks

## Requirements
- Requirements are available at the `requirements.txt` file. For installation, use the command: `pip install -r requirements.txt`. Used Python: 3.10.12.

## File Structure

- `main.py`: Orchestrates the entire pruning, fine-tuning, and evaluation pipeline
- `amp_criterion.py`: Implements the AMP method for measuring component importance
- `prune_groups.py`: Handles the actual pruning of model components
- `fine_tune.py`: Fine-tunes the pruned model using LoRA

## Usage

### Basic Usage

```bash
python main.py --model_path "meta-llama/Llama-2-7b-hf" --arch llama --criterion standard --iteration 10 --save_dir output --gpu 0
```

### Parameters

- `--model_path`: Path or name of the pre-trained model (e.g., "meta-llama/Llama-2-7b-hf")
- `--arch`: Model architecture (currently only "llama" is supported)
- `--criterion`: Pruning criterion ("standard", "reversed", or "random")
  - `standard`: Prunes components with lowest importance scores
  - `reversed`: Prunes components with highest importance scores
  - `random`: Prunes components randomly
- `--iteration`: Iteration number for pruning (higher values result in more aggressive pruning) (*).
- `--save_dir`: Directory to save outputs
- `--gpu`: GPU ID to use

(*) Each iteration number corresponds to pruning a parameter count similar to that of an entire layer of the LLM. For example, if you use --iteration 10, you will get a model equivalent to pruning 10 layers. For LLaMA-2 7B, each layer corresponds to approximately 3% of the total parameters in the model. This implementation was designed to enable better comparisons with depth-pruning methods.

## Individual Components

### AMP Criterion

The `amp_criterion.py` file implements the core AMP method:

```bash
python amp_criterion.py --model_name "meta-llama/Llama-2-7b-hf" --arch llama --task heads --max_prompts 50
```

### Pruning

To run only the pruning step:

```bash
python prune_groups.py --model_path "meta-llama/Llama-2-7b-hf" --save_dir output --arch llama --criterion standard --iteration 5
```

### Fine-tuning

To fine-tune a pruned model:

```bash
python fine_tune.py --pruned_model_dir output/prune_iter_5 --tuned_model_dir output/tuned_iter_5 --arch llama --num_train_epochs 2
```

### Evaluation

To evaluate a fine-tuned model:

```bash
lm_eval --model hf --model_args pretrained=output/tuned_iter_5 --tasks piqa,hellaswag,winogrande,arc_easy,arc_challenge,wikitext --output_path output/eval_iter_5 --trust_remote_code
```

## How It Works

1. **Importance Measurement**: The code measures the importance of attention heads and MLP neurons by analyzing their activation patterns during inference on a set of prompts.

2. **Pruning**: Based on the importance scores, the least important components are identified and removed from the model.

3. **Fine-tuning**: The pruned model is fine-tuned using LoRA (Low-Rank Adaptation) to recover performance.

4. **Evaluation**: The pruned and fine-tuned model is evaluated on standard benchmarks like PIQA, HellaSwag, Winogrande and ARC.

## Extending to Other Architectures

The code is designed to be easily extended to other model architectures. To add support for a new architecture:

1. Add the architecture settings to the `ARCHITECTURE_SETTINGS` dictionary in both `amp_criterion.py` and `prune_groups.py`
2. Update the architecture-specific functions in `prune_groups.py` to handle the new architecture
3. Add the appropriate target modules for LoRA fine-tuning in `fine_tune.py`


## Acknowledgements
We thank Instituto de Ciência e Tecnologia Itaú (ICTi) for the technical support, resources, and financial aid in the development of the research project. The authors would also like to thank the Programa de Bolsas Itaú (PBI) of the Centro de Ciência de Dados (C2D), supported by Itaú Unibanco S.A.

## Publication
Please cite [our paper](https://arxiv.org/abs/2504.21174) in your publications if it helps your research.
```bash
@inproceedings{Mugnaini:2025,
author    = {Leandro Giusti Mugnaini, 
			Bruno Lopes Yamamoto, 
			Lucas Lauton de Alcantara, 
			Victor Zacarias, 
			Edson Bollis, 
			Lucas Pellicer, 
			Anna Helena Reali Costa and
			Artur Jordao},
title     = {Efficient LLMs with AMP: Attention Heads and MLP Pruning},
booktitle = {International Joint Conference on Neural Networks (IJCNN).},
}
```
