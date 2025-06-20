import os
import os
from glob import glob
import argparse
import torch
from Aposcore.inference_dataset import get_mdn_score
from Aposcore.Aposcore import Aposcore

import pymol2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ApoScore scoring on ligand and receptor files."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="fegrow_result",
        help="Input folder containing ligand and receptor files.",
    )
    parser.add_argument(
        "--ligand_pattern",
        type=str,
        default="cs_optimised*",
        help="Glob pattern for ligand files.",
    )
    parser.add_argument(
        "--receptor_pattern",
        type=str,
        default="rec_final*",
        help="Glob pattern for receptor files.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints/ApoScore_time_split_0.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on."
    )
    parser.add_argument(
        "--dis_threshold",
        type=float,
        default=5.0,
        help="Distance threshold for scoring.",
    )
    return parser.parse_args()


def collect_files(folder, pattern):
    files = glob(os.path.join(folder, pattern))
    if not files:
        print(f"Warning: No files found for pattern {pattern} in {folder}")
    # Remove files which have zero size
    files = [f for f in files if os.path.getsize(f) > 0]
    if not files:
        print(f"Warning: All files for pattern {pattern} in {folder} are empty.")
    return files

# remove hydrogen from receptor as required by get_mdn_score(()
def strip_hydrogens_from_receptors(receptor_files):
    for rec_path in receptor_files:
        # Skip if noH version already exists
        if rec_path.endswith("-noH.pdb"):
            continue
        noH_path = rec_path.replace(".pdb", "-noH.pdb")
        with pymol2.PyMOL() as pymol:
            pymol.cmd.load(rec_path, "rec")
            pymol.cmd.remove("hydrogens")
            pymol.cmd.save(noH_path, "rec")

def get_scores():
    args = parse_args()
    # Initialize model
    model_mdn = Aposcore(
        35,
        hidden_dim=256,
        num_heads=4,
        dropout=0.1,
        crossAttention=True,
        atten_active_fuc="softmax",
        num_layers=6,
        interact_type="product",
    )
    ligs = sorted(collect_files(args.input_folder, args.ligand_pattern))
    raw_receptors = sorted(collect_files(args.input_folder, args.receptor_pattern))
    
    # Strip hydrogens before scoring
    strip_hydrogens_from_receptors(raw_receptors)
    # Use the -noH versions for scoring
    receptors = [r.replace(".pdb", "-noH.pdb") for r in raw_receptors if r.endswith(".pdb")]


    if not ligs or not receptors:
        print("No ligand or receptor files found. Exiting.")
        return
    scores = get_mdn_score(
        ligs,
        receptors,
        model_mdn,
        args.ckpt,
        args.device,
        dis_threshold=args.dis_threshold,
    )
    # Convert scores to a dictionary of dict[(Path,Path) float], (where the keys are ligand and receptor file paths)
    scores = {
        (lig, rec): score for (lig, rec), score in zip(zip(ligs, receptors), scores)
    }
    # Sort by score - highest first
    scores = {
        (lig, rec): score
        for (lig, rec), score in sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )
    }
    # Save scores to a file
    output_file = os.path.join(args.input_folder, "scores.txt")
    with open(output_file, "w") as f:
        for lig, score in scores.items():
            f.write(f"{lig}: {score}\n")
    print("Scores:", scores)


if __name__ == "__main__":
    get_scores()
