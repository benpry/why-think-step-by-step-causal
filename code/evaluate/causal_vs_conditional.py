"""
Compute differences between causal and conditional rpobabilities for pairs of variables.
"""
import pickle
import numpy as np
import pandas as pd
from pyprojroot import here
from argparse import ArgumentParser
from itertools import product
import sys
from pgmpy.inference.ExactInference import VariableElimination

sys.path.extend(["./core", "../core", "./code/core"])

from bayes_net import pom_to_pgm

parser = ArgumentParser()
# parser.add_argument("--n_nets", type=str, default=100)
parser.add_argument(
    "--bayes_net_file",
    type=str,
    default="data/bayes_nets/nets_n-100_nodes-100_edges-100.pkl",
)

selected_nets = [2, 28, 33, 51, 6, 61, 64, 68, 81, 9]

if __name__ == "__main__":
    args = parser.parse_args()
    nets = pickle.load(open(here(args.bayes_net_file), "rb"))
    for net in nets:
        net.from_pickle()

    marginal_vs_conditional = []
    conditional_vs_intervention = []
    for net_i in selected_nets:
        net = nets[net_i]
        model = pom_to_pgm(net.model)

        df_pairs = pd.read_csv(
            here(f"data/training-data/selected-pairs/selected-pairs-net-{net_i}.csv")
        )
        for index, row in df_pairs.iterrows():
            var1, var2 = row["var1"], row["var2"]

            for condition, target in ((var1, var2), (var2, var1)):
                for condition_val in (True, False):
                    VE = VariableElimination(model)
                    marginal = VE.query([target]).values[1]
                    conditional_prob = VE.query(
                        [target], evidence={condition: condition_val}
                    ).values[1]
                    VE_do = VariableElimination(model.do(condition))
                    intervention_prob = VE_do.query(
                        [target], evidence={condition: condition_val}
                    ).values[1]

                    print(f"p({target}) = {marginal}")
                    print(
                        f"p({target}|{condition}={int(condition_val)}) = {conditional_prob}"
                    )
                    print(
                        f"p({target}|do({condition}={int(condition_val)})) = {intervention_prob}"
                    )
                    marginal_vs_conditional.append(abs(conditional_prob - marginal))
                    conditional_vs_intervention.append(
                        abs(conditional_prob - intervention_prob)
                    )

    print(
        f"Mean difference between marginal and conditional: {np.mean(marginal_vs_conditional)}"
    )
    print(
        f"Mean difference between conditional and intervention: {np.mean(conditional_vs_intervention)}"
    )
