import argparse
import json
import os
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

def make_dataset(num_samples, num_classes, num_features, Wstar=None, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    X = np.random.randn(num_samples, num_features)
    if Wstar is None:
        Wstar = np.random.randn(num_features, num_classes)  # ground truth classifier
    y = np.argmax(X @ Wstar, axis=1)  # labels
    return X, y, Wstar, len(np.unique(y)) == num_classes  # Check if all classes appear

def train_step(W, X, y_one_hot, learning_rate):
    def loss_fn(W, X, y):
        preds = jnp.dot(X, W)
        return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(preds), axis=1))
    
    grads = jax.grad(loss_fn)(W, X, y_one_hot)
    W -= learning_rate * grads
    return W

def compute_accuracy(W, X, y):
    preds = jnp.argmax(jnp.dot(X, W), axis=1)
    # Ensure y is a one-dimensional JAX array for comparison
    y = jnp.asarray(y).squeeze() if y.ndim > 1 else jnp.asarray(y)
    return jnp.mean(preds == y)

def frobenius_norm(W):
    return jnp.sqrt(jnp.sum(jnp.square(W)))

def run_experiment(params):
    X, y, Wstar, _ = make_dataset(params['num_samples'], params['num_classes'], 
                                  params['num_features'], random_seed=params['random_seed_for_make_data'])
    encoder = OneHotEncoder()
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1)).toarray()
    key = jax.random.PRNGKey(params['random_seed_for_W_init'])
    W = jax.random.normal(key, (params['num_features'], params['num_classes']))
    
    all_results = {
        "accuracy_train": [],
        "accuracy_test": [],
        "loss": []
    }

    for epoch in range(params['num_epochs']):
        W = train_step(W, X, y_one_hot, params['learning_rate'])
        if epoch % params['record_every_n_epochs'] == 0:
            loss = jax.jit(jax.vmap(lambda x, y: -jnp.sum(y * jax.nn.log_softmax(jnp.dot(x, W), axis=-1), axis=-1)))(X, y_one_hot).mean()
            accuracy = compute_accuracy(W, X, y)
            all_results["accuracy_train"].append(float(accuracy))
            all_results["loss"].append(float(loss))

    final_infos = {
        "exp_with_var_lr": {
            "means": {
                "accuracy_train_mean": np.mean(all_results["accuracy_train"]),
                "loss_mean": np.mean(all_results["loss"])
            }
        }
    }
    
    return all_results, final_infos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_features", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--random_seed_for_make_data", type=int, default=42)
    parser.add_argument("--random_seed_for_W_init", type=int, default=0)
    parser.add_argument("--record_every_n_epochs", type=int, default=10)
    args = parser.parse_args()

    params = {
        'num_samples': args.num_samples,
        'num_classes': args.num_classes,
        'num_features': args.num_features,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'random_seed_for_make_data': args.random_seed_for_make_data,
        'random_seed_for_W_init': args.random_seed_for_W_init,
        'record_every_n_epochs': args.record_every_n_epochs
    }

    all_results, final_infos = run_experiment(params)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)

if __name__ == "__main__":
    main()
