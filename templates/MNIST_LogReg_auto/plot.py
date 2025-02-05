import os
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

folders = [f for f in os.listdir('.') if "run_" in f and osp.isdir(f)]
labels = {"run_0": "Baseline",}

for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        results_dict = np.load(osp.join(folder, "all_results.npy"), allow_pickle=True).item()
        for metric, values in results_dict.items():
            plt.plot(range(len(values)), values[metric], label=labels.get(folder, "Run"), color=np.random.rand(3,))
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.title(f"{metric} over Epochs")
            plt.savefig(f"{metric}_{folder}.png")
        plt.show()