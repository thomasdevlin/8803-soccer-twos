import os

import gym
from gym_unity.envs import ActionFlattener
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import AgentInterface


class SpaceOnlyEnv(gym.Env):
    """Minimal env shell used to construct RLlib policy objects."""

    observation_space = None
    action_space = None

    def __init__(self, _config=None):
        self.observation_space = self.__class__.observation_space
        self.action_space = self.__class__.action_space

    def reset(self):
        return self.observation_space.sample()

    def step(self, _action):
        return self.observation_space.sample(), 0.0, True, {}


class TeamAgent(AgentInterface):
    """
    An agent definition for policies trained with DQN on `team_vs_policy` variation with `single_player=True`.
    """

    def __init__(self, env):
        # find RLlib checkpoint from checkpoint directory
        self.flattener = ActionFlattener(env.action_space.nvec)

        checkpoint_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoint_000100"
        )
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint-100")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

        # if not ray.is_initialized():
        #     ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        SpaceOnlyEnv.observation_space = env.observation_space
        SpaceOnlyEnv.action_space = self.flattener.action_space

        # create ray tune agent from loaded checkpoint 
        self.trainer = PPOTrainer(
            env=SpaceOnlyEnv,
            config={
                "num_workers": 0,
                "num_gpus": 0,
                "framework": "torch",
            },
        )
        self.trainer.restore(checkpoint_path)

    def act(self, observation):
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        # for each team player
        for player_id in observation:
            # feed observation into loaded model and update action dictionary
            action = self.trainer.compute_action(observation[player_id], explore=False)
            if isinstance(action, (int, np.integer)):
                actions[player_id] = self.flattener.lookup_action(int(action))
            else:
                actions[player_id] = action
        return actions
