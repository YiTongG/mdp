# utils.py
import re
from decimal import Decimal
import argparse
import sys

comment = re.compile(r"^#.*$")
reward_line = re.compile(r"^[a-zA-Z0-9\-']+ *= *-?\d*\.?\d*$")
edge_line = re.compile(r"^[a-zA-Z0-9\-']+ *: *\[( *[a-zA-Z0-9\-']+ *,?)* *]$")
probability_line = re.compile(r"^[a-zA-Z0-9\-']+ *% *( *\d*\.?\d*,?)* *$")


def tokenize_reward(line):
    match = re.match(r"([^=]+)=(.+)", line)
    if match:
        return match.group(1).strip(), Decimal(match.group(2).strip())
    return None


def tokenize_edge(line):
    match = re.match(r"([^:]+):\s*\[([^\]]+)\]", line)
    if match:
        return match.group(1).strip(), [x.strip() for x in match.group(2).split(',')]
    return None


def tokenize_probability(line):
    match = re.match(r"([^%]+)%(.+)", line)
    if match:
        return match.group(1).strip(), [x.strip() for x in match.group(2).split()]
    return None


def tokenize(line):
    tokenize_functions = {'=': tokenize_reward, ':': tokenize_edge, '%': tokenize_probability}
    for key, func in tokenize_functions.items():
        result = func(line)
        if result:
            return result
    return None


def parse_arguments():
    parser = argparse.ArgumentParser(description='Markov Process Solver')
    parser.add_argument('-df', nargs='?', type=float, required=False, default=1.0)
    parser.add_argument('-min', required=False, action='store_true')
    parser.add_argument('-tol', nargs='?', default=0.01, type=float, required=False)
    parser.add_argument('-iter', nargs='?', default=100, type=float, required=False)
    parser.add_argument('filename')

    return parser.parse_args(sys.argv[1:])
