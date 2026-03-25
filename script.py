import argparse
import os

import numpy as np
from tensorflow import keras

GRID_SIZE   = 4
ACTION_SIZE = 4
MOVES = {
    0: (-1,  0),
    1: ( 1,  0),
    2: ( 0, -1),
    3: ( 0,  1),
}


class GridWorld:
    def __init__(self):
        self.reset()

    def reset(self):
        self.agent_pos    = (0, 0)
        self.goal_pos     = (3, 3)
        self.obstacle_pos = (1, 1)
        return self.get_state()

    def get_state(self):
        state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        state[self.agent_pos] = 1.0
        return state.flatten()

    def step(self, action):
        x, y   = self.agent_pos
        dx, dy = MOVES[action]
        nx, ny = x + dx, y + dy

        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            self.agent_pos = (nx, ny)

        if self.agent_pos == self.goal_pos:
            return self.get_state(), 10, True
        if self.agent_pos == self.obstacle_pos:
            return self.get_state(), -5, False
        return self.get_state(), -1, False


def select_action(model, state):
    q_values = model.predict(np.array([state], dtype=np.float32), verbose=0)
    return int(np.argmax(q_values[0]))


def evaluate(model, episodes=20, max_steps=50):
    env      = GridWorld()
    rewards  = []
    successes = 0

    for ep in range(1, episodes + 1):
        state        = env.reset()
        total_reward = 0
        reached_goal = False

        for _ in range(max_steps):
            action              = select_action(model, state)
            state, reward, done = env.step(action)
            total_reward       += reward
            if done:
                reached_goal = True
                break

        rewards.append(total_reward)
        if reached_goal:
            successes += 1

        print(
            f"Épisode {ep:3d}/{episodes} | "
            f"Récompense: {total_reward:6.1f} | "
            f"Objectif atteint: {'Oui' if reached_goal else 'Non'}"
        )

    print("\nRésumé")
    print(f"Récompense moyenne :  {np.mean(rewards):.2f}")
    print(f"Meilleure récompense : {np.max(rewards):.2f}")
    print(f"Pire récompense :      {np.min(rewards):.2f}")
    print(f"Taux de succès :       {100.0 * successes / episodes:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tester un modèle Double DQN entraîné.")
    parser.add_argument("--model",     default="double_dqn_model.keras", help="Chemin vers le fichier .keras")
    parser.add_argument("--episodes",  type=int, default=20, help="Nombre d'épisodes d'évaluation")
    parser.add_argument("--max-steps", type=int, default=50, help="Nombre maximum de pas par épisode")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Fichier modèle introuvable : {args.model}")

    model = keras.models.load_model(args.model)
    evaluate(model, episodes=args.episodes, max_steps=args.max_steps)