import os
import os
from glob import glob
import argparse
import torch
from Aposcore.inference_dataset import get_mdn_score
from Aposcore.Aposcore import Aposcore

import pymol2
import re

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

    receptor_files = sorted(collect_files(args.input_folder, args.receptor_pattern))
    ligand_files = sorted(collect_files(args.input_folder, "*.sdf"))

    strip_hydrogens_from_receptors(receptor_files)
    receptor_noH_files = [r.replace(".pdb", "-noH.pdb") for r in receptor_files]

    receptor_map = {}
    for rec in receptor_noH_files:
        match = re.search(r"rec_final_(\d+)-noH\.pdb", os.path.basename(rec))
        if match:
            receptor_map[match.group(1)] = rec

    ligand_map = {}
    for lig in ligand_files:
        match = re.search(r"rec_(\d+)_mol\d+\.sdf", os.path.basename(lig))
        if match:
            rec_index = match.group(1)
            ligand_map.setdefault(rec_index, []).append(lig)

    output_lines = []
    all_scores = []  # To track all scored pairs

    for rec_index, rec_path in receptor_map.items():
        ligands = ligand_map.get(rec_index, [])
        if not ligands:
            continue

        output_lines.append(f"\n=== Scores for {os.path.basename(rec_path)} ===\n")

        for lig_path in sorted(ligands):
            score = get_mdn_score(
                [lig_path],
                [rec_path],
                model_mdn,
                args.ckpt,
                args.device,
                dis_threshold=args.dis_threshold,
            )[0]

            output_lines.append(f"{os.path.basename(lig_path)}: {score:.4f}")
            all_scores.append((lig_path, rec_path, score))

    # Sort by score descending
    top_scores = sorted(all_scores, key=lambda x: x[2], reverse=True)[:3]

    # Write scores
    output_file = os.path.join(args.input_folder, "scores_per_receptor.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))

        f.write("\n\n=== Top 3 Overall Scoring Pairs ===\n")
        for i, (lig, rec, score) in enumerate(top_scores, start=1):
            f.write(
                f"{i}. Ligand: {os.path.basename(lig)} | "
                f"Receptor: {os.path.basename(rec)} | "
                f"Score: {score:.4f}\n"
            )

    # Print top 3
    print(f"\nScoring complete. Output saved to {output_file}")
    print("\n=== Top 3 Overall Scoring Pairs ===")
    for i, (lig, rec, score) in enumerate(top_scores, start=1):
        print(
            f"{i}. Ligand: {os.path.basename(lig)} | "
            f"Receptor: {os.path.basename(rec)} | "
            f"Score: {score:.4f}"
        )
if __name__ == "__main__":
    get_scores()