"""Federated learning with manual-feature DQN client selection and PSO aggregation."""

from __future__ import annotations

import argparse
import importlib
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

import metrics.writer as metrics_writer

from baseline_constants import (
    ACCURACY_KEY,
    BYTES_READ_KEY,
    BYTES_WRITTEN_KEY,
    LOCAL_COMPUTATIONS_KEY,
    MAIN_PARAMS,
    MODEL_PARAMS,
)
from client import Client
from dqn_pso_selector import SmartDQNSelector
from dqn_pso_utils import Particle, initialize_gbest, run_pso_iterations
from server import Server
from utils.model_utils import read_data

STAT_METRICS_PATH = "metrics/stat_metrics.csv"
SYS_METRICS_PATH = "metrics/sys_metrics.csv"


def parse_dqn_pso_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", type=str, required=True)
    parser.add_argument("-model", type=str, required=True)
    parser.add_argument("--num-rounds", type=int, default=-1)
    parser.add_argument("--eval-every", type=int, default=-1)
    parser.add_argument("--clients-per-round", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-name", type=str, default="metrics")
    parser.add_argument("--metrics-dir", type=str, default="metrics")
    parser.add_argument("--use-val-set", action="store_true")
    parser.add_argument("-t", type=str, default="large")
    parser.add_argument("-lr", type=float, default=-1)

    parser.add_argument("--pso-iterations", type=int, default=5)
    parser.add_argument("--pso-inertia", type=float, default=0.5)
    parser.add_argument("--pso-local-acc", type=float, default=0.7)
    parser.add_argument("--pso-global-acc", type=float, default=1.4)
    parser.add_argument("--dqn-epsilon", type=float, default=0.3)
    parser.add_argument("--dqn-epsilon-min", type=float, default=0.05)
    parser.add_argument("--dqn-epsilon-decay", type=float, default=0.995)
    parser.add_argument("--dqn-replay-start", type=int, default=32)
    parser.add_argument("--dqn-batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--val-max-samples", type=int, default=50)

    return parser.parse_args()


def setup_clients(dataset: str, model, use_val_set: bool = False) -> List[Client]:
    eval_set = "test" if not use_val_set else "val"
    train_data_dir = os.path.join("..", "data", dataset, "data", "train")
    test_data_dir = os.path.join("..", "data", dataset, "data", eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    if len(groups) == 0:
        groups = [[] for _ in users]
    return [
        Client(u, g, train_data[u], test_data[u], model)
        for u, g in zip(users, groups)
    ]


def get_stat_writer_function(ids, groups, num_samples, args):
    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round,
            ids,
            metrics,
            groups,
            num_samples,
            partition,
            args.metrics_dir,
            "{}_{}".format(args.metrics_name, "stat"),
        )

    return writer_fn


def get_sys_writer_function(args):
    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round,
            ids,
            metrics,
            groups,
            num_samples,
            "train",
            args.metrics_dir,
            "{}_{}".format(args.metrics_name, "sys"),
        )

    return writer_fn


def print_metrics(metrics, weights, prefix=""):
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print(
            "%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g"
            % (
                prefix + metric,
                np.average(ordered_metric, weights=ordered_weights),
                np.percentile(ordered_metric, 10),
                np.percentile(ordered_metric, 50),
                np.percentile(ordered_metric, 90),
            )
        )


def print_stats(num_round, server, clients, num_samples, args, writer, use_val_set):
    train_stat_metrics = server.test_model(clients, set_to_use="train")
    print_metrics(train_stat_metrics, num_samples, prefix="train_")
    writer(num_round, train_stat_metrics, "train")

    eval_set = "test" if not use_val_set else "val"
    test_stat_metrics = server.test_model(clients, set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, prefix="{}_".format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)


def build_validation_data(
    clients: Sequence[Client],
    val_fraction: float,
    max_per_client: int,
    seed: int,
) -> Dict[str, List]:
    rng = random.Random(seed)
    val_x: List = []
    val_y: List = []
    for client in clients:
        data = client.train_data
        total = len(data["y"])
        if total == 0:
            continue
        target = max(1, int(total * val_fraction))
        target = min(target, max_per_client, total)
        indices = rng.sample(range(total), target)
        val_x.extend([data["x"][i] for i in indices])
        val_y.extend([data["y"][i] for i in indices])
    return {"x": val_x, "y": val_y}


def compute_diversities(clients: Sequence[Client], num_classes: int) -> List[float]:
    diversities = []
    for client in clients:
        labels = np.array(client.train_data["y"])
        if labels.size == 0:
            diversities.append(0.0)
            continue
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=-1)
        counts = np.bincount(labels.astype(int), minlength=num_classes)
        total = float(np.sum(counts))
        if total == 0.0:
            diversities.append(0.0)
            continue
        probs = counts / total
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        diversities.append(float(entropy / np.log(num_classes)))
    return diversities


def infer_num_classes(clients: Sequence[Client]) -> int:
    max_label = -1
    for client in clients:
        labels = np.array(client.train_data["y"])
        if labels.size == 0:
            continue
        if labels.ndim > 1:
            return int(labels.shape[-1])
        max_label = max(max_label, int(np.max(labels)))
    if max_label < 0:
        raise ValueError("Unable to infer num_classes from empty training data.")
    return max_label + 1


def main() -> None:
    args = parse_dqn_pso_args()

    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)

    model_path = "%s/%s.py" % (args.dataset, args.model)
    if not os.path.exists(model_path):
        raise ValueError("Please specify a valid dataset and a valid model.")
    module_path = "%s.%s" % (args.dataset, args.model)

    print("############################## %s ##############################" % module_path)
    mod = importlib.import_module(module_path)
    ClientModel = getattr(mod, "ClientModel")

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    tf.logging.set_verbosity(tf.logging.WARN)

    model_params = MODEL_PARAMS[module_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    tf.reset_default_graph()
    client_model = ClientModel(args.seed, *model_params)
    server = Server(client_model)

    clients = setup_clients(args.dataset, client_model, args.use_val_set)
    client_ids = [c.id for c in clients]
    client_groups = {c.id: c.group for c in clients}
    client_num_samples = {c.id: c.num_samples for c in clients}
    print("Clients in Total: %d" % len(clients))
    clients_per_round = min(clients_per_round, len(clients))

    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)

    print("--- Random Initialization ---")
    print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

    val_data = build_validation_data(
        clients, args.val_fraction, args.val_max_samples, args.seed
    )
    if len(val_data["y"]) == 0:
        raise ValueError("Validation data is empty. Check dataset preprocessing.")

    num_classes = infer_num_classes(clients)
    client_diversities = compute_diversities(clients, num_classes)

    dqn_selector = SmartDQNSelector(
        num_clients=len(clients),
        num_to_select=clients_per_round,
        state_size=3,
        epsilon=args.dqn_epsilon,
        epsilon_min=args.dqn_epsilon_min,
        epsilon_decay=args.dqn_epsilon_decay,
        seed=args.seed,
    )
    dqn_selector.set_diversities(client_diversities)

    particles = [
        Particle.from_weights(i, client_model.get_params()) for i in range(len(clients))
    ]

    global_weights = [w.copy() for w in client_model.get_params()]
    previous_val_accuracy = 0.0

    for round_num in range(num_rounds):
        print(
            "--- Round %d of %d: Training %d Clients ---"
            % (round_num + 1, num_rounds, clients_per_round)
        )

        local_losses: List[float] = []
        local_updates: List[List[np.ndarray]] = []
        local_computations: Dict[str, float] = {}
        for client in clients:
            client.model.set_params(global_weights)
            comp, num_samples, update = client.train(
                num_epochs=args.num_epochs, batch_size=args.batch_size
            )
            client.model.set_params(update)
            metrics = client.test("train")
            local_losses.append(float(metrics["loss"]))
            local_updates.append([layer.copy() for layer in update])
            local_computations[client.id] = comp

        dqn_selector.client_losses = np.array(local_losses, dtype=np.float32)
        selected_client_indices, current_state = dqn_selector.select_clients(
            clients_per_round
        )
        selected_particles = [particles[i] for i in selected_client_indices]

        for idx, particle in zip(selected_client_indices, selected_particles):
            particle.current_weights = [layer.copy() for layer in local_updates[idx]]

        gbest_weights, gbest_score = initialize_gbest(
            client_model, selected_particles, val_data
        )
        gbest_weights, gbest_score = run_pso_iterations(
            client_model,
            selected_particles,
            val_data,
            gbest_weights,
            gbest_score,
            iterations=args.pso_iterations,
            inertia=args.pso_inertia,
            local_acc=args.pso_local_acc,
            global_acc=args.pso_global_acc,
        )

        global_weights = [layer.copy() for layer in gbest_weights]
        client_model.set_params(global_weights)
        server.model = [layer.copy() for layer in global_weights]

        val_metrics = client_model.test(val_data)
        val_acc = float(val_metrics[ACCURACY_KEY])
        reward = (val_acc - previous_val_accuracy) * 100.0

        selected_losses = [local_losses[i] for i in selected_client_indices]
        dqn_selector.update_metrics(selected_client_indices, selected_losses, reward)
        next_state = dqn_selector.get_state()
        dqn_selector.remember(current_state, selected_client_indices, reward, next_state)
        if len(dqn_selector.memory) >= args.dqn_replay_start:
            dqn_selector.replay(batch_size=args.dqn_batch_size)

        previous_val_accuracy = val_acc

        sys_metrics = {
            client_ids[i]: {
                BYTES_READ_KEY: client_model.size,
                BYTES_WRITTEN_KEY: client_model.size,
                LOCAL_COMPUTATIONS_KEY: local_computations[client_ids[i]],
            }
            for i in selected_client_indices
        }
        sys_writer_fn(round_num + 1, [client_ids[i] for i in selected_client_indices], sys_metrics, client_groups, client_num_samples)

        if (round_num + 1) % eval_every == 0 or (round_num + 1) == num_rounds:
            print_stats(
                round_num + 1,
                server,
                clients,
                client_num_samples,
                args,
                stat_writer_fn,
                args.use_val_set,
            )

    dqn_selector.close()
    server.close_model()


if __name__ == "__main__":
    main()
