from typing import Optional
import gymnasium as gym
from functools import partial # Import partial for cleaner thunk creation

# Make sure these imports are correct based on your project structure
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import FrameStack # Moved from inside function
# Assuming CPUGymWrapper and RecordEpisode are in mani_skill.utils.wrappers
from mani_skill.utils.wrappers import CPUGymWrapper
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list[gym.Wrapper] = [],
    # --- Added arguments ---
    save_trajectory: bool = False,
    trajectory_name: str = "trajectory",
    # --- End Added arguments ---
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
        save_trajectory: whether to save the trajectories during evaluation recording.
        trajectory_name: the name to use for saved trajectory files.
    """
    # Get obs_horizon safely
    obs_horizon = other_kwargs.get("obs_horizon", 1)

    if sim_backend == "physx_cpu":

        # Modified helper function signature
        def cpu_make_env_thunk(
            env_id,
            seed,
            video_dir_i=None, # Renamed for clarity
            env_kwargs=dict(),
            obs_horizon=1, # Pass directly
            wrappers_list=[], # Pass directly
            # --- Added arguments for helper ---
            save_trajectory_flag=False,
            trajectory_name_str="trajectory",
            # --- End added arguments ---
        ):
            env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
            for wrapper in wrappers_list:
                env = wrapper(env)
            env = FrameStack(env, num_stack=obs_horizon) # Use passed obs_horizon
            env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

            # Get max_episode_steps for RecordEpisode
            max_episode_steps = env_kwargs.get("max_episode_steps")
            if max_episode_steps is None:
                 try:
                     max_episode_steps = env.spec.max_episode_steps
                 except AttributeError:
                     print("Warning: max_episode_steps not found for RecordEpisode in CPU env, using default 200.")
                     max_episode_steps = 200

            if video_dir_i:
                env = RecordEpisode(
                    env,
                    output_dir=video_dir_i,
                    # --- Use passed arguments ---
                    save_trajectory=save_trajectory_flag,
                    trajectory_name=trajectory_name_str,
                    # --- End Use passed arguments ---
                    info_on_video=True,
                    # source_type="diffusion_policy", # Keep or remove as needed
                    # source_desc="diffusion_policy evaluation rollout", # Keep or remove
                    max_steps_per_video=max_episode_steps, # Added for CPU path consistency
                    video_fps=30, # Added for CPU path consistency
                )
            # Seeding (important to do it before returning the env)
            env.reset(seed=seed) # Use reset for seeding in newer gym versions
            env.action_space.seed(seed)
            # env.observation_space.seed(seed) # Obs space seeding often not needed/supported like this
            return env


        # Determine vector environment class
        vector_cls = (
            gym.vector.SyncVectorEnv
            if num_envs == 1
            # Use forkserver context for multiprocessing safety with some libraries
            else partial(gym.vector.AsyncVectorEnv, context="forkserver")
        )

        # Create the list of thunks, passing necessary args including the new ones
        env_thunks = [
            partial( # Use partial for cleaner argument passing
                cpu_make_env_thunk,
                env_id=env_id,
                seed=seed,
                # Only record video for the first environment in CPU mode
                video_dir_i=f"{video_dir}/env_{seed}" if video_dir and seed == 0 else None,
                env_kwargs=env_kwargs,
                obs_horizon=obs_horizon,
                wrappers_list=wrappers,
                # Pass the new arguments
                save_trajectory_flag=save_trajectory if seed == 0 else False, # Only save traj for env 0
                trajectory_name_str=trajectory_name,
            )
            for seed in range(num_envs)
        ]
        env = vector_cls(env_thunks)

    # GPU backend path
    else:
        # Assume gym.make handles num_envs for GPU backends like Isaac Gym
        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1, # Often needed for randomization in ManiSkill
            **env_kwargs
        )
        # Find max_episode_steps *after* making the env
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)

        # Apply custom wrappers
        for wrapper in wrappers:
            env = wrapper(env)

        # Apply FrameStack
        env = FrameStack(env, num_stack=obs_horizon)

        # Apply RecordEpisode if video_dir is provided
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir, # Saves to subdirs env_0, env_1 etc. automatically
                # --- Use passed arguments ---
                save_trajectory=save_trajectory,
                trajectory_name=trajectory_name,
                # --- End Use passed arguments ---
                save_video=True, # Assuming video is always wanted if dir provided
                # source_type="diffusion_policy", # Keep or remove
                # source_desc="diffusion_policy evaluation rollout", # Keep or remove
                max_steps_per_video=max_episode_steps,
                video_fps=30, # Added for consistency
            )

        # Apply ManiSkillVectorEnv wrapper (usually last)
        # ignore_terminations should likely be False for standard evaluation
        # unless you have a specific reason to ignore them (like in PPO training rollouts)
        env = ManiSkillVectorEnv(env, ignore_terminations=False, record_metrics=True)

    return env