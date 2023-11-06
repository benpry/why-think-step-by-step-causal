"""
This file computes the true conditional probabilities in a Bayes net.
"""
import pandas as pd
import numpy as np
import pickle
from itertools import product
from argparse import ArgumentParser
import sys
from pgmpy.inference.ExactInference import VariableElimination

sys.path.extend(["../core", "./code/core"])
from utils import pom_to_pgm
from pyprojroot import here


def get_probs_from_samples(samples, target_var, condition_var, condition_val):
    condition_samples = [s for s in samples if s[condition_var] == condition_val]
    target_prob = np.mean([s[target_var] for s in condition_samples])

    # get the difference between the true marginal and the conditional
    marginal = np.mean([s[target_var] for s in samples])

    return target_prob, marginal


def get_probs_analytically_with_intervention(
    model, target_var, condition_var, condition_val
):
    conditional_inference = VariableElimination(model)
    conditional_prob = conditional_inference.query(
        [target_var], evidence={condition_var: condition_val}
    ).values[1]
    causal_inference = VariableElimination(model.do(condition_var))
    intervention_prob = conditional_inference.query(
        [target_var], evidence={condition_var: condition_val}
    ).values[1]

    # get the difference between the true marginal and the conditional

    return marginal, conditional_prob, intervention_prob


parser = ArgumentParser()
parser.add_argument("--net_idx", type=int)
parser.add_argument("--bayes-net-file", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    nets = pickle.load(open(here(args.bayes_net_file), "rb"))
    net = nets[args.net_idx]
    net.from_pickle()
    net.node_order = None
    all_vars = net.graph.nodes
    net_pgm = pom_to_pgm(net.model)

    distances = net.get_distances()

    rows = []
    for target, condition in product(all_vars, all_vars):
        if target == condition:
            continue

        if target in distances[condition]:
            distance = int(distances[condition][target])
        else:
            distance = -1

        for condition_val in (False, True):
            (
                marginal,
                conditional_prob,
                intervention_prob,
            ) = get_probs_analytically_with_intervention(
                net, target, condition, condition_val
            )
            condition_influence = abs(conditional_prob - marginal)

            rows.append(
                {
                    "target_var": target,
                    "condition_var": condition,
                    "condition_val": condition_val,
                    "cond_target_dist": distance,
                    "conditional_prob": conditional_prob,
                    "marginal_prob": marginal,
                    "intervention_prob": intervention_prob,
                    "condition_influence": condition_influence,
                }
            )

    df_true = pd.DataFrame(rows)
    df_true.to_csv(
        here(
            f"data/evaluation/causal/true-probs/true-probabilities-net-{args.net_idx}.csv"
        )
    )
