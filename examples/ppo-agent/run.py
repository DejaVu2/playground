"""Implementation of a simple deterministic agent using Docker."""

import numpy as np
from ppo_agent import *
import tensorflow as tf
from pommerman.runner import DockerAgentRunner


class PPOAgent(BaseAgent):
    """The PPOAgent. Acts through the algorith, not here."""

    def __init__(self, character=characters.Bomber):
        super(PPOAgent, self).__init__(character)

    def act(self, obs, action_space):
        pass


class MyPPOAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        self.stackedobs = np.zeros((1, 11, 11, 72))

    def act(self, observation, action_space):
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        obs = docker_featurize(observation)
        self.stackedobs[..., -obs.shape[-1]:] = obs
        action, _, _, _ = self.model.step(self.stackedobs)
        return int(action)

    def initialize(self, model):
        self.model = model


def main():
    '''Inits and runs a Docker Agent'''
    agent_list = [
        PPOAgent(),
        PPOAgent(),
        PPOAgent(),
        PPOAgent(),
    ]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config):
        model = learn(network='cnn',
                      env=make_vec_env(agent_list, 4),
                      nsteps=8192,
                      nminibatches=1024,
                      log_interval=1,
                      ent_coef=0.01,
                      lr=lambda _: 2e-4,
                      cliprange=lambda _: 0.1,
                      # total_timesteps=int(1e7),
                      total_timesteps=int(0),
                      save_interval=10,
                      load_path='00030',
                      )

        agent = MyPPOAgent()
        agent.initialize(model)
        agent.run()


if __name__ == "__main__":
    main()
