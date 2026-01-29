"""DQN-based client selector for federated learning."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

State = np.ndarray
Action = List[int]
Transition = Tuple[State, Action, float, State]


class SmartDQNSelector:
    """DQN agent that selects clients using manual features.

    Features per client:
        1) diversity (fixed)
        2) normalized loss (dynamic)
        3) contribution score (dynamic, EMA)
    """

    def __init__(
        self,
        num_clients: int,
        num_to_select: int,
        state_size: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 0.3,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 0,
    ) -> None:
        self.num_clients = num_clients
        self.num_to_select = num_to_select
        self.state_per_client = state_size
        self.input_shape = (num_clients * state_size,)
        self.action_size = num_clients

        self.memory: Deque[Transition] = deque(maxlen=2000)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.client_diversity = np.zeros(num_clients, dtype=np.float32)
        self.client_losses = np.ones(num_clients, dtype=np.float32) * 10.0
        self.client_contributions = np.zeros(num_clients, dtype=np.float32)

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.session = tf.Session(graph=self.graph)
            tf.keras.backend.set_session(self.session)
            self.model = self._build_model()
            self.session.run(tf.global_variables_initializer())

    def _build_model(self) -> tf.keras.Model:
        """Build the DQN network."""
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(
                128, input_shape=self.input_shape, activation="relu"
            )
        )
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
        )
        return model

    def set_diversities(self, diversities: Sequence[float]) -> None:
        """Set fixed diversity features per client."""
        if len(diversities) != self.num_clients:
            raise ValueError("Diversities length must match num_clients.")
        self.client_diversity = np.array(diversities, dtype=np.float32)

    def update_metrics(
        self, selected_indices: Sequence[int], losses: Sequence[float], reward: float
    ) -> None:
        """Update per-client loss and contribution metrics."""
        if len(selected_indices) != len(losses):
            raise ValueError("selected_indices and losses must have the same length.")
        for idx, loss in zip(selected_indices, losses):
            self.client_losses[idx] = float(loss)
            self.client_contributions[idx] = (
                0.9 * self.client_contributions[idx] + 0.1 * reward
            )

    def get_state(self) -> State:
        """Build a state vector from current client metrics."""
        avg_loss = float(np.mean(self.client_losses))
        std_loss = float(np.std(self.client_losses)) + 1e-9
        normalized_losses = (self.client_losses - avg_loss) / std_loss

        state: List[float] = []
        for i in range(self.num_clients):
            state.append(float(self.client_diversity[i]))
            state.append(float(normalized_losses[i]))
            state.append(float(self.client_contributions[i]))

        return np.array(state, dtype=np.float32).reshape(1, -1)

    def select_clients(self, num_to_select: int) -> Tuple[Action, State]:
        """Select clients using epsilon-greedy policy."""
        state = self.get_state()
        selected_indices = self.select_action(state, self.epsilon, num_to_select)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return selected_indices, state

    def select_action(
        self, state: State, epsilon: float, num_to_select: int
    ) -> Action:
        """Select actions from a given state."""
        if num_to_select > self.num_clients:
            raise ValueError("num_to_select cannot exceed num_clients.")

        if np.random.rand() <= epsilon:
            sorted_indices_by_loss = np.argsort(self.client_losses)[::-1]
            candidate_pool_size = min(
                self.num_clients, max(self.num_clients // 2, num_to_select)
            )
            candidates = sorted_indices_by_loss[:candidate_pool_size]
            return random.sample(list(candidates), num_to_select)

        with self.graph.as_default():
            tf.keras.backend.set_session(self.session)
            q_values = self.model.predict(state, verbose=0)[0]
        selected_indices = np.argsort(q_values)[-num_to_select:]
        return selected_indices.tolist()

    def remember(self, state: State, action: Action, reward: float, next_state: State) -> None:
        """Store a transition in replay memory."""
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size: int) -> None:
        """Train the DQN from replay memory."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([x[0] for x in minibatch])
        next_states = np.vstack([x[3] for x in minibatch])

        with self.graph.as_default():
            tf.keras.backend.set_session(self.session)
            target = self.model.predict(states, verbose=0)
            target_next = self.model.predict(next_states, verbose=0)

            for i, (_, action, reward, _) in enumerate(minibatch):
                target_val = reward + self.gamma * np.amax(target_next[i])
                for client_idx in action:
                    target[i][client_idx] = target_val

            self.model.fit(states, target, epochs=1, verbose=0)

    def close(self) -> None:
        """Close the DQN session."""
        with self.graph.as_default():
            self.session.close()
