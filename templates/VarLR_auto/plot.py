import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import random

def random_color():
    return "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])

def load_and_plot_results(folders):
    labels = {"run_0": "Baseline",}
    
    for folder in folders:
        if folder.startswith("run") and osp.isdir(folder):
            results_dict = np.load(osp.join(folder, "all_results.npy"), allow_pickle=True).item()
            for metric, values in results_dict.items():
                iteration_number = list(range(len(values)))
                plt.plot(iteration_number, values, label=labels[folder], color=random_color())
    
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Experiment Results Over Epochs")
    plt.legend()
    plt.show()
    plt.savefig("experiment_results.png")

folders = [d for d in os.listdir(".") if osp.isdir(d)]
load_and_plot_results(folders)