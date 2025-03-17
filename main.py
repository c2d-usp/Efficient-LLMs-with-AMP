#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def create_directories(save_dir, iteration):
    """
    Creates and returns directories for logs, pruned model, tuned model, and evaluation outputs.
    The directory names include the iteration information.
    """
    outputs_dir = os.path.join(save_dir, "logs")
    pruned_model_dir = os.path.join(save_dir, f"prune_iter_{iteration}")
    tuned_model_dir = os.path.join(save_dir, f"tuned_iter_{iteration}")
    eval_dir = os.path.join(save_dir, f"eval_iter_{iteration}")

    for d in [outputs_dir, pruned_model_dir, tuned_model_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)

    return outputs_dir, pruned_model_dir, tuned_model_dir, eval_dir


def run_command(command, log_file):
    """
    Runs a subprocess command, logging both stdout and stderr to the provided log_file.
    Exits if the command returns a non-zero exit code.
    """
    with open(log_file, "a") as lf:
        lf.write(f"\n[COMMAND] {' '.join(command)}\n")
        process = subprocess.run(command, stdout=lf, stderr=subprocess.STDOUT)
        if process.returncode != 0:
            sys.exit(f"Command {' '.join(command)} failed. Check {log_file} for details.")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate the pruning, fine-tuning, and evaluation pipeline."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Name or path of the base model")
    parser.add_argument("--arch", type=str, choices=["llama"], required=True, help="Model architecture")
    parser.add_argument("--criterion", type=str, choices=["standard", "reversed", "random"], default="standard", help="Pruning criterion")
    parser.add_argument("--iteration", type=int, default=10, help="Iteration number for pruning")
    parser.add_argument("--save_dir", type=str, default="output", help="Base directory for all outputs")
    parser.add_argument("--gpu", type=str, default="1", help="GPU id to use (e.g., '0' or '1')")
    args = parser.parse_args()

    outputs_dir, pruned_model_dir, tuned_model_dir, eval_dir = create_directories(args.save_dir, args.iteration)
    log_file = os.path.join(outputs_dir, f"log_iter_{args.iteration}.txt")

    with open(log_file, "a") as lf:
        lf.write("[START] Orchestrating pipeline\n")

    # 1. Pruning Step
    with open(log_file, "a") as lf:
        lf.write("[STEP 1] Starting pruning process...\n")

    prune_command = [
        "python", "prune_groups.py",
        "--model_path", args.model_path,
        "--save_dir", args.save_dir,
        "--arch", args.arch,
        "--criterion", args.criterion,
        "--iteration", str(args.iteration)
    ]
    run_command(prune_command, log_file)

    with open(log_file, "a") as lf:
        lf.write(f"[STEP 1] Pruning completed. Pruned model saved at {pruned_model_dir}\n")

    # 2. Fine-Tuning Step
    with open(log_file, "a") as lf:
        lf.write("[STEP 2] Starting fine-tuning process...\n")

    finetune_command = [
        "python", "fine_tune.py",
        "--pruned_model_dir", pruned_model_dir,
        "--tuned_model_dir", tuned_model_dir,
        "--arch", args.arch,
    ]
    run_command(finetune_command, log_file)

    with open(log_file, "a") as lf:
        lf.write(f"[STEP 2] Fine-tuning completed. Tuned model saved at {tuned_model_dir}\n")


    # 3. Evaluation Step
    with open(log_file, "a") as lf:
        lf.write("[STEP 3] Starting evaluation process...\n")

    eval_command = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={tuned_model_dir}",
        "--tasks", "piqa,hellaswag,winogrande,arc_easy,arc_challenge,wikitext",
        "--output_path", eval_dir,
        "--trust_remote_code"
    ]
    run_command(eval_command, log_file)

    with open(log_file, "a") as lf:
        lf.write(f"[STEP 3] Evaluation completed. Results saved at {eval_dir}\n")
        lf.write("[END] Pipeline finished successfully.\n")


if __name__ == "__main__":
    main()
