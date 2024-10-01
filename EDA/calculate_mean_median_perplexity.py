import json

import numpy as np

with open("/home/ambaranov/paper-2024/EDA/results_pos.json") as f:
    res_pos = json.load(f)

print("res_pos", "mean", np.mean(res_pos["perplexities"]), "median", np.median(res_pos["perplexities"]))

with open("/home/ambaranov/paper-2024/EDA/results_neg.json") as f:
    res_neg = json.load(f)

print("res_neg", "mean", np.mean(res_neg["perplexities"]), "median", np.median(res_neg["perplexities"]))

with open("/home/ambaranov/paper-2024/EDA/results_perplexity_additional.json") as f:
    res_ria = json.load(f)


print("res_ria", "mean", np.mean(res_ria["perplexities"]), "median", np.median(res_ria["perplexities"]))
