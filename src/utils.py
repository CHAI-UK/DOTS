import numpy as np
import uuid
import pandas as pd
import os 
from itertools import permutations, product

def num_errors(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i+1:], order[i]].sum()
    return err
    
def fullAdj2Order(A):
    order = list(A.sum(axis=1).argsort())
    order.reverse()
    return order

def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A

def get_transitive_closure_from_orderings(topological_orderings):
    full_DAGs = [full_DAG(ordering) for ordering in topological_orderings]
    transitive_closure = np.logical_and.reduce(full_DAGs).astype(int)
    return transitive_closure, full_DAGs

def get_soft_transitive_closure_from_orderings(topological_orderings):
    full_DAGs = [full_DAG(ordering) for ordering in topological_orderings]
    transitive_closure_soft = np.sum(full_DAGs, axis=0).astype(int) > 0
    return transitive_closure_soft

def np_to_csv(array, save_path):
    """
    Convert np array to .csv
    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    #output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')
    output = os.path.join(save_path, 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output


def get_value_from_str(exp_name : str, variable : str, type_func = str):
    value_init_pos = exp_name.rfind(variable) + len(variable)
    if exp_name.rfind(variable) == -1:
        return np.nan 
    new_str = exp_name[value_init_pos:]
    end_pos = new_str.find("_") 
    if end_pos == -1:
        return type_func(new_str)
    else:
        return type_func(new_str[:end_pos])

# The function all_possible_permutations generates all possible permutations of the flattened list 
# where elements are only permuted within their respective inner lists.
def all_possible_permutations(list_of_lists):
    # Generate all permutations for each inner list
    perms = [list(permutations(inner_list)) for inner_list in list_of_lists]
    # Compute the Cartesian product of these permutations
    product_of_perms = product(*perms)
    # Flatten each combination of permutations and collect the results
    result = [ [item for perm in perm_tuple for item in perm] for perm_tuple in product_of_perms ]
    return result

# create a block diagonal matrix with block size 3 of the same size as true_causal_matrix
def get_instantaneous_effect_matrix_constraint(nb_timesteps, nb_variables):
    return 1- np.kron(np.eye(nb_timesteps), np.ones((nb_variables, nb_variables)))

def get_temporal_effect_matrix_constraint(nb_timesteps, nb_variables):
    return 1 - np.triu(np.ones((nb_variables*nb_timesteps, nb_variables*nb_timesteps)), k=1)

def apply_temporal_constraint(adj, nb_timesteps, nb_variables, instantanous_constraint: bool = True):
    constraint_matrix_temp = get_temporal_effect_matrix_constraint(nb_timesteps=nb_timesteps, nb_variables=nb_variables)
    # get intersectio between constraint_matrix_inst and constraint_matrix_temp
    if instantanous_constraint:
        constraint_matrix_inst = get_instantaneous_effect_matrix_constraint(nb_timesteps=nb_timesteps, nb_variables=nb_variables)
        constraint_all = np.logical_and(constraint_matrix_inst, constraint_matrix_temp)
    else:
        constraint_all = constraint_matrix_temp

    adj_constrained = np.logical_and(adj, constraint_all).astype(int)
    return adj_constrained