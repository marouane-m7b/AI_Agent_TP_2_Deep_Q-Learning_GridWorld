import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

GRID_SIZE           = 4
STATE_SIZE          = GRID_SIZE * GRID_SIZE
ACTION_SIZE         = 4
GAMMA               = 0.9
LEARNING_RATE       = 0.01
EPSILON             = 1.0
EPSILON_MIN         = 0.01
EPSILON_DECAY       = 0.995
BATCH_SIZE          = 32
MEMORY_SIZE         = 2000
EPISODES            = 1000
UPDATE_TARGET_EVERY = 10

MOVES = {
    0: (-1,  0),
    1: ( 1,  0),
    2: ( 0, -1),
    3: ( 0,  1),
}


class GridWorld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        self.agent_pos    = (0, 0)
        self.goal_pos     = (3, 3)
        self.obstacle_pos = (1, 1)
        return self.get_state()

    def get_state(self):
        state = np.zeros((GRID_SIZE, GRID_SIZE))
        state[self.agent_pos] = 1
        return state.flatten()

    def step(self, action):
        x, y   = self.agent_pos
        dx, dy = MOVES[action]
        nx, ny = x + dx, y + dy

        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            self.agent_pos = (nx, ny)

        if self.agent_pos == self.goal_pos:
            return self.get_state(), 10, True
        elif self.agent_pos == self.obstacle_pos:
            return self.get_state(), -5, False
        else:
            return self.get_state(), -1, False


class DoubleDQNAgent:
    def __init__(self):
        self.state_size  = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.memory      = deque(maxlen=MEMORY_SIZE)
        self.epsilon     = EPSILON

        self.model        = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear'),
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch       = random.sample(self.memory, BATCH_SIZE)
        states      = np.array([t[0] for t in batch])
        next_states = np.array([t[3] for t in batch])

        q_online_current = self.model.predict(states,      verbose=0)
        q_online_next    = self.model.predict(next_states, verbose=0)
        q_target_next    = self.target_model.predict(next_states, verbose=0)

        targets = q_online_current.copy()

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                targets[i][action] = reward
            else:
                best_action        = np.argmax(q_online_next[i])
                targets[i][action] = reward + GAMMA * q_target_next[i][best_action]

        self.model.fit(states, targets, epochs=1, batch_size=BATCH_SIZE, verbose=0)

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY


env    = GridWorld()
agent  = DoubleDQNAgent()
scores = []

print("Entraînement du Double DQN sur GridWorld 4x4...\n")

for episode in range(EPISODES):
    state        = env.reset()
    total_reward = 0

    for step in range(50):
        action                   = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state        = next_state
        total_reward += reward
        if done:
            break

    agent.replay()
    scores.append(total_reward)

    if (episode + 1) % UPDATE_TARGET_EVERY == 0:
        agent.update_target_network()

    if (episode + 1) % 50 == 0:
        avg = np.mean(scores[-50:])
        print(f"Épisode {episode+1:4d}/{EPISODES} | "
              f"Score moyen (50 derniers): {avg:6.1f} | "
              f"Epsilon: {agent.epsilon:.4f}  "
              f"{'✅ Apprentissage réussi!' if avg > -20 else ''}")
    else:
        print(f"Épisode {episode+1:4d}/{EPISODES} | "
              f"Score: {total_reward:6.1f} | "
              f"Epsilon: {agent.epsilon:.4f}")

agent.model.save("double_dqn_model.keras")
print("\nTerminé ! Modèle sauvegardé → double_dqn_model.keras")