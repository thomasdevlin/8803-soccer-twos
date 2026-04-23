from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos
import numpy as np


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class ShapingWrapper(gym.core.Wrapper, MultiAgentEnv):
    "An RLLib wrapper which shapes rewards"
    def __init__(self, env):
        super().__init__(env)

        # measured field geometry from Pitch.fbx mesh
        self.field_geometry = {
            "length": 30.0,
            "width": 0.0,
            "goal_a_x": -15.0,
            "goal_b_x": 15.0,
            "goal_y": 0.0,
            "goal_width": 8.0,
            "corner_radius": 3.5
        }
        self.objective_weights = {
            "offensive": 0.00001,
            "defensive": 0.00001,
            "possession": 0.0001,
            "heading": 0.0001,
        }

        self.my_team = None
    
    def step(self, action):
        obs, rewards, terminateds, infos = self.env.step(action)
        for id in obs:
            if self.my_team is None:
                if id < 2:
                    self.my_team = "a"
                    self.field_geometry["opp_goal_pos"] = np.array([self.field_geometry["goal_b_x"],self.field_geometry["goal_y"]])
                    self.field_geometry["own_goal_pos"] = np.array([self.field_geometry["goal_a_x"],self.field_geometry["goal_y"]])
                else:
                    self.my_team = "b"
                    self.field_geometry["opp_goal_pos"] = np.array([self.field_geometry["goal_a_x"],self.field_geometry["goal_y"]])
                    self.field_geometry["own_goal_pos"] = np.array([self.field_geometry["goal_b_x"],self.field_geometry["goal_y"]])

            # see soccer_twos env wrappers.py line 291-307
            agent_pos = np.array(infos[id]["player_info"]["position"])
            ball_pos = np.array(infos[id]["ball_info"]["position"])

            ball_dist_to_opp_goal = np.linalg.norm(agent_pos-self.field_geometry["opp_goal_pos"])
            ball_dist_to_own_goal = np.linalg.norm(agent_pos-self.field_geometry["own_goal_pos"])
            ball_dist_to_own_goal = np.pow(ball_dist_to_own_goal, 2) # encourage stronger defense when closer to own goal
            agent_dist_to_ball = np.linalg.norm(agent_pos-ball_pos)
            # agent_heading_to_ball = np.arctan2(ball_pos[1]-agent_pos[1], ball_pos[0]-agent_pos[0]) - infos[id]["player_info"]["rotation_y"]


            rewards[id] = (rewards[id] 
                + self.objective_weights["offensive"] * ball_dist_to_opp_goal 
                - self.objective_weights["defensive"] * ball_dist_to_own_goal 
                - self.objective_weights["possession"] * agent_dist_to_ball
                # + self.objective_weights["heading"] * agent_heading_to_ball
            )

        return obs, rewards, terminateds, infos


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)
    # return ShapingWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s