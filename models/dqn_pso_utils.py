"""PSO utilities for federated learning."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Particle:
    """Particle for PSO-based aggregation."""

    particle_id: int
    current_weights: List[np.ndarray]
    velocities: List[np.ndarray]
    pbest_weights: List[np.ndarray]
    pbest_score: float

    @classmethod
    def from_weights(cls, particle_id: int, initial_weights: Sequence[np.ndarray]) -> "Particle":
        velocities = [np.random.rand(*layer.shape) / 5 - 0.10 for layer in initial_weights]
        weights_copy = [layer.copy() for layer in initial_weights]
        return cls(
            particle_id=particle_id,
            current_weights=weights_copy,
            velocities=velocities,
            pbest_weights=[layer.copy() for layer in initial_weights],
            pbest_score=float("inf"),
        )

    def update_velocity_and_position(
        self,
        gbest_weights: Sequence[np.ndarray],
        inertia: float,
        local_acc: float,
        global_acc: float,
    ) -> None:
        """Update velocity and weights based on PSO dynamics."""
        new_weights = [None] * len(self.current_weights)
        local_rand, global_rand = random.random(), random.random()

        for i, layer_weights in enumerate(self.current_weights):
            new_velocity = (
                inertia * self.velocities[i]
                + local_acc * local_rand * (self.pbest_weights[i] - layer_weights)
                + global_acc * global_rand * (gbest_weights[i] - layer_weights)
            )
            self.velocities[i] = new_velocity
            new_weights[i] = layer_weights + new_velocity
        self.current_weights = new_weights


def evaluate_weights(
    model,
    weights_list: Sequence[Sequence[np.ndarray]],
    val_data: dict,
) -> List[float]:
    """Evaluate a list of weight sets on the validation data."""
    losses: List[float] = []
    for weights in weights_list:
        model.set_params(weights)
        metrics = model.test(val_data)
        losses.append(float(metrics["loss"]))
    return losses


def initialize_gbest(
    model,
    particles: Sequence[Particle],
    val_data: dict,
) -> Tuple[List[np.ndarray], float]:
    """Initialize gbest by evaluating current particles."""
    weights_list = [p.current_weights for p in particles]
    losses = evaluate_weights(model, weights_list, val_data)
    best_idx = int(np.argmin(losses))
    gbest_score = float(losses[best_idx])
    gbest_weights = [layer.copy() for layer in weights_list[best_idx]]

    for particle, loss, weights in zip(particles, losses, weights_list):
        particle.pbest_score = float(loss)
        particle.pbest_weights = [layer.copy() for layer in weights]

    return gbest_weights, gbest_score


def run_pso_iterations(
    model,
    particles: Sequence[Particle],
    val_data: dict,
    gbest_weights: Sequence[np.ndarray],
    gbest_score: float,
    iterations: int,
    inertia: float,
    local_acc: float,
    global_acc: float,
) -> Tuple[List[np.ndarray], float]:
    """Run PSO iterations to refine global weights."""
    for _ in range(iterations):
        for particle in particles:
            particle.update_velocity_and_position(
                gbest_weights, inertia, local_acc, global_acc
            )

        weights_list = [p.current_weights for p in particles]
        losses = evaluate_weights(model, weights_list, val_data)

        for particle, loss in zip(particles, losses):
            if loss < particle.pbest_score:
                particle.pbest_score = float(loss)
                particle.pbest_weights = [layer.copy() for layer in particle.current_weights]
            if loss < gbest_score:
                gbest_score = float(loss)
                gbest_weights = [layer.copy() for layer in particle.current_weights]

    return [layer.copy() for layer in gbest_weights], float(gbest_score)
