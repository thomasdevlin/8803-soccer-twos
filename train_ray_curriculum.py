import os
import yaml

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from TEAM1_AGENT import TeamAgent as Team1BaselineAgent
from ceia_baseline_agent import RayAgent
from gym_unity.envs import ActionFlattener
from soccer_twos import EnvType

from utils import create_rllib_env, sample_pos_vel, sample_player


NUM_ENVS_PER_WORKER = 1
CURRICULUM_FILE = "curriculum.yaml"
# CURRICULUM_FILE = "curriculum_ceia_test.yaml"
BASE_PORT = 5021

current = 0
ceia_baseline_agent = None
team1_baseline_agent = None
final_baseline_next = None
baseline_action_lookup = {
    tuple(int(v) for v in values): key
    for key, values in ActionFlattener([3, 3, 3]).action_lookup.items()
}
with open(CURRICULUM_FILE) as f:
    curriculum = yaml.load(f, Loader=yaml.FullLoader)
tasks = curriculum["tasks"]


def get_ceia_opponent_policy(env):
    global ceia_baseline_agent
    if ceia_baseline_agent is None:
        ceia_baseline_agent = RayAgent(env)

    def ceia_policy(obs):
        raw_action = ceia_baseline_agent.act({0: obs})[0]
        return to_discrete_action(raw_action)

    return ceia_policy


def get_team1_opponent_policy(env):
    global team1_baseline_agent
    if team1_baseline_agent is None:
        team1_baseline_agent = Team1BaselineAgent(env)

    def team1_policy(obs):
        raw_action = team1_baseline_agent.act({0: obs})[0]
        return to_discrete_action(raw_action)

    return team1_policy


def to_discrete_action(action):
    if isinstance(action, (int, np.integer)):
        return int(action)

    if isinstance(action, np.ndarray):
        values = tuple(int(v) for v in action.tolist())
    elif isinstance(action, (list, tuple)):
        values = tuple(int(v) for v in action)
    else:
        return int(action)

    if values in baseline_action_lookup:
        return baseline_action_lookup[values]

    raise ValueError("Could not map baseline action {} to discrete index".format(values))


def apply_task_config(env, config_name):
    if config_name == "none":
        return

    if config_name == "random_players":
        env.set_policies(lambda *_: env.action_space.sample())
        return

    # Backward compatible alias: disable old self_play behavior.
    if config_name == "self_play":
        return

    if config_name == "ceia_baseline":
        env.set_opponent_policy(get_ceia_opponent_policy(env))
        if hasattr(env, "set_teammate_policy"):
            env.set_teammate_policy(lambda *_: 0)
        return

    if config_name == "team1_baseline":
        env.set_opponent_policy(get_team1_opponent_policy(env))
        if hasattr(env, "set_teammate_policy"):
            env.set_teammate_policy(lambda *_: 0)
        return

    raise KeyError("Unknown config_fn: {}".format(config_name))


class CurriculumUpdateCallback(DefaultCallbacks):
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ) -> None:
        global current, tasks, final_baseline_next

        config_name = tasks[current]["config_fn"]
        if config_name in ("team1_baseline", "ceia_baseline"):
            if final_baseline_next is None:
                final_baseline_next = config_name
            config_name = final_baseline_next
            final_baseline_next = (
                "ceia_baseline"
                if config_name == "team1_baseline"
                else "team1_baseline"
            )
        else:
            final_baseline_next = None

        for env in base_env.get_unwrapped():
            apply_task_config(env, config_name)
            env.env_channel.set_parameters(
                ball_state=sample_pos_vel(tasks[current]["ranges"]["ball"]),
                players_states={
                    player: sample_player(tasks[current]["ranges"]["players"][player])
                    for player in tasks[current]["ranges"]["players"]
                },
            )

    def on_train_result(self, **info):
        global current

        if info["result"]["episode_reward_mean"] > 1.5:
            if current < len(tasks) - 1:
                print("---- Updating tasks!!! ----")
                current += 1
                print(f"Current task: {current} - {tasks[current]['name']}")


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_curriculum",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 6,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CurriculumUpdateCallback,
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "flatten_branched": True,
                "single_player": True,
                "opponent_policy": lambda *_: 0,
                "base_port": BASE_PORT,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 200,
            "batch_mode": "truncate_episodes",
            "train_batch_size": 5000,
            "lr": 0.0003,                       # hyperparameters from pugliese paper
            "gamma": 0.99,
            "lambda": 0.95,
            # "train_batch_size": 4_000,
            # "sgd_minibatch_size": 256,
            # "num_sgd_iter": 10,
            "clip_param":0.2,
        },
        stop={
            "timesteps_total": 20000000,
            "time_total_s": 36000, # 10h
            # "episode_reward_mean": 1.99,
            "episode_reward_mean": 10.0,
        },
        # stop={
        #     "timesteps_total": 15000000,
        #     "time_total_s": 7200, # 2h
        #     "episode_reward_mean": 1.9,
        # },
        checkpoint_freq=5,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_curriculum/PPO_Soccer_5103e_00000_0_2026-04-23_16-23-40/checkpoint_000010/checkpoint-10",
        # restore="~/scratch/8803-soccer-twos/ray_results/PPO_curriculum/PPO_Soccer_f8069_00000_0_2026-04-23_16-56-58/checkpoint_000055/checkpoint-55",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
