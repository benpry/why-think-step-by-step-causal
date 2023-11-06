"""
This file contains functions that take in samples from Bayes nets and turn them into GPT-2 prompts.
"""
import numpy as np


def initialize_var_order_and_n_vars(sample, var_order, n_vars):
    # if the number of variables wasn't specified, generate all the variables
    if n_vars is None and var_order is None:
        n_vars = len(sample)
    elif n_vars is None:
        n_vars = len(var_order)

    # if no variable order is specified, generate in a random order
    if var_order is None:
        var_order = list(
            np.random.choice(list(sample.keys()), size=n_vars, replace=False)
        )

    return var_order, n_vars


def create_target_sample_str(sample, target_var, var_order=None, n_vars=None):
    var_order, n_vars = initialize_var_order_and_n_vars(sample, var_order, n_vars)

    # make sure the target variable is last
    if target_var in var_order:
        var_order.remove(target_var)
    else:
        var_order.pop(-1)

    var_order.append(target_var)

    sample_str = f"###\ntarget: {target_var}\n"

    # add the rest of the variables
    for var in var_order:
        sample_str += f"{var}={sample[var]}\n"

    return sample_str


def create_simple_sample_str(sample, var_order=None, n_vars=None):
    var_order, n_vars = initialize_var_order_and_n_vars(sample, var_order, n_vars)

    sample_str = "###\n"
    for var in var_order:
        sample_str += f"{var}={sample[var]}\n"

    return sample_str


def create_partial_overlap_sample_str(
    sample, var_order=None, n_vars=None, pairs_to_avoid=()
):
    var_order, n_vars = initialize_var_order_and_n_vars(sample, var_order, n_vars)
    # make sure none of the pairs to avoid are present
    for var1 in var_order:
        for var2 in var_order:
            if (var1, var2) in pairs_to_avoid:
                var_order.remove(var2)

    sample_str = f"###\ntarget: {var_order[-1]}\n"
    for var in var_order:
        sample_str += f"{var}={sample[var]}\n"

    return sample_str
