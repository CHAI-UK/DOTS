import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from src.dots import DOTS
from src.eval_utils import perform_eval, read_json

# load data
data = pd.read_csv("./data/samples.csv")
data_lagged = pd.read_csv("./data/samples_lagged.csv")

nb_vars = data.shape[1]
nb_nodes = data_lagged.shape[1]
nb_lags = nb_nodes // nb_vars

# train DOTS
dots = DOTS(nb_nodes)
adj = dots.fit(data_lagged.to_numpy(), nb_timesteps=nb_lags, nb_variables=nb_vars)

df_adj = pd.DataFrame(adj, columns=data_lagged.columns, index=data_lagged.columns)

# load ground truth
graphs_true = read_json("./data/graphs_true.json")

# evaluate
result = perform_eval(graphs_true, df_adj, data.columns)
print(result)