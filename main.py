import os
import time

import gymnasium as gym
from env import CircuitEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

def train(circuit):
    circuit_file_name = circuit + '.trk'

    log_path = os.path.join('Training', 'Logs')
    env = CircuitEnv(circuit_name=circuit_file_name, zoom=5.0, render_mode="human")
    env = DummyVecEnv([lambda:env])
    #check_env(env, True, False)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)

    model_file_name = 'ppo_model_' + circuit
    ppo_path = os.path.join('models', model_file_name)
    model.save(ppo_path)

    env.close()

def test(circuit):
    circuit_file_name = circuit + '.trk'
    env = CircuitEnv(circuit_name=circuit_file_name, zoom=4.0, render_mode="human")

    obs, info = env.reset()

    episodes = 5
    for episode in range(1, episodes+1):
        terminated = False
        truncated = False
        score = 0

        while not terminated and not truncated:
            action = [0.1, 0.0] #env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward

            print(obs, reward)
    
        if terminated or truncated:
            obs, info = env.reset()

        print('Episode:{} Score:{}'.format(episode, score))

    env.close()


def evaluate(circuit):
    model_file_name = 'ppo_model_' + circuit
    ppo_path = os.path.join('models', model_file_name)

    circuit_file_name = circuit + '.trk'
    env = CircuitEnv(circuit_name=circuit_file_name, zoom=4.0, render_mode="human")
    env = DummyVecEnv([lambda:env])
    model = PPO.load(ppo_path, env=env)

    evaluate_policy(model, env, n_eval_episodes=10, render=True)

    env.close()


if __name__ == '__main__':

    train('silverstone')
