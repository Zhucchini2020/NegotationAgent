#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH -J PrimoBot_train
#SBATCH -o output_primo_train.out

source /gpfs/data/epavlick/etan13/NegotationAgent/negotiate_env/Scripts/activate
python3 deeprlearning.py
