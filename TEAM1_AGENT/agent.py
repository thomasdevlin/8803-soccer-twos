import os
from typing import Dict

import gym
from gym_unity.envs import ActionFlattener
import numpy as np
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./ray_results/PPO_curriculum/PPO_Soccer_f8069_00000_0_2026-04-23_16-56-58/checkpoint_000055/checkpoint-55",
)
POLICY_NAME = "default"  # this may be useful when training with selfplay
MODEL_CONFIG = {
    "vf_share_layers": True,
    "fcnet_hiddens": [256, 256],
    "fcnet_activation": "relu",
}
_SHARED_TRAINER = None


class SpaceOnlyEnv(gym.Env):
    """Minimal env that only exposes spaces for RLlib policy construction."""

    observation_space = None
    action_space = None

    def __init__(self, config=None):
        _ = config
        self.observation_space = self.__class__.observation_space
        self.action_space = self.__class__.action_space

    def reset(self):
        return self.observation_space.sample()

    def step(self, _action):
        return self.observation_space.sample(), 0.0, True, {}


def create_space_only_env(_config=None):
    return SpaceOnlyEnv(_config)


class TeamAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        global _SHARED_TRAINER

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.name = "team1_agent"

        self.flattener = None
        policy_action_space = env.action_space
        if hasattr(env.action_space, "nvec"):
            self.flattener = ActionFlattener(env.action_space.nvec)
            policy_action_space = self.flattener.action_space

        # Provide spaces expected by RLlib without creating another Unity env.
        SpaceOnlyEnv.observation_space = env.observation_space
        SpaceOnlyEnv.action_space = policy_action_space
        tune.registry.register_env("DummyEnv", create_space_only_env)

        # Build a minimal restore config without unpickling params.pkl.
        config = {
            "env": "DummyEnv",
            "framework": "torch",
            "num_workers": 0,
            "num_gpus": 0,
            "model": MODEL_CONFIG,
        }

        if _SHARED_TRAINER is None:
            cls = get_trainable_cls(ALGORITHM)
            agent = cls(env=config["env"], config=config)
            agent.restore(CHECKPOINT_PATH)
            _SHARED_TRAINER = agent

        self.trainer = _SHARED_TRAINER

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
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
            if self.flattener is not None and isinstance(action, (int, np.integer)):
                actions[player_id] = self.flattener.lookup_action(int(action))
            else:
                actions[player_id] = action
        return actions
