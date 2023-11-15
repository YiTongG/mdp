# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# main.py
import argparse
import sys
from mdp import MDP
from utils import parse_arguments

if __name__ == '__main__':
    params = parse_arguments()
    mdp = MDP.read_inputfile(params.filename, df=params.df, tol=params.tol, max_iter=params.iter, use_min=params.min)
    mdp.mdp_solver()
    mdp.mdp_result()

