# ALGO_NAME = "BC_Diffusion_rgbd_UNet" # Removed as it's not used

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional
import json # Added for reading demo metadata

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate # evaluate function assumes agent, env, device etc.
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    evaluate: bool = False # Added
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None # Added
    """path to a pretrained checkpoint file to start evaluation/training from"""

    env_id: str = "PegInsertionSide-v1"
    """the id of the environment"""
    demo_path: str = (
        "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    )
    """the path of demo dataset, it is expected to be a ManiSkill dataset h5py format file"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""


    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 64  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )  # default setting is about ~4.5M params
    n_groups: int = (
        8  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    )

    # Environment/experiment specific arguments
    obs_mode: str = "rgb+depth"
    """The observation mode to use for the environment, which dictates what visual inputs to pass to the model. Can be "rgb", "depth", or "rgb+depth"."""
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


class SmallDemoDataset_DiffusionPolicy(Dataset):  # Load everything into memory
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, device, num_traj, obs_horizon, pred_horizon, control_mode):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        from diffusion_policy.utils import load_demo_dataset
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        # trajectories['observations'] is a list of dict, each dict is a traj, with keys in obs_space, values with length L+1
        # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(
                obs_traj_dict, obs_space
            )  # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.Tensor(
                    _obs_traj_dict["depth"].astype(np.float32)
                ).to(device=device, dtype=torch.float16)
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(
                    device
                )  # still uint8
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(
                device
            )
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())
        # Pre-process the actions
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i]).to(
                device=device
            )
        print(
            "Obs/action pre-processing is done, start to pre-compute the slice indices..."
        )

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
            "delta_pos" in control_mode
            or control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            print("Detected a delta controller type, padding with a zero action to ensure the arm stays still after solving tasks.")
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device
            )
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            # NOTE for absolute joint pos control probably should pad with the final joint position action.
            raise NotImplementedError(f"Control Mode {control_mode} not supported")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon

        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]  # start+self.obs_horizon is at least 1
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:  # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert (
            len(env.single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (
            env.single_action_space.low == -1
        ).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        self.visual_encoder = PlainConv(
            in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

    def encode_obs(self, obs_seq, eval_mode):
        # Convert incoming obs to torch tensors if they are numpy arrays (common during inference)
        if isinstance(obs_seq["state"], np.ndarray):
            for k, v in obs_seq.items():
                obs_seq[k] = torch.from_numpy(v).to(device=self.noise_pred_net.device) # Assume net is on correct device

        img_seq = None
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            if img_seq is None:
                 img_seq = depth
            elif self.include_rgb: # Concatenate only if both are present
                img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k

        if img_seq is None:
            raise ValueError("Agent requires either RGB or Depth observations.")

        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq["state"].shape[0]

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(
            obs_seq, eval_mode=False
        )  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=action_seq.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=action_seq.device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad() # Decorator for no_grad during inference

# (ensure other necessary imports like nn are present in the class scope)

# Inside the Agent class definition:

    def get_action(self, obs_seq_input): # Renamed input variable
        """
        Predicts the action sequence based on the input observation sequence.

        Args:
            obs_seq_input (dict): A dictionary containing observation sequences.
                                Expected keys depend on obs_mode (e.g., 'state', 'rgb', 'depth').
                                Values are expected to be numpy arrays initially (B, T, ...),
                                but this method handles if they are already tensors.

        Returns:
            np.ndarray: The predicted action sequence for the action horizon (B, act_horizon, act_dim).
        """
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters) # Not needed if using DDPM and same steps

        # Use a key known to exist, like 'state', to get batch size
        # Ensure 'state' key exists and handle potential errors if needed
        if "state" not in obs_seq_input:
            raise ValueError("Input observation sequence dictionary must contain 'state' key.")
        B = obs_seq_input["state"].shape[0] # Batch size is num_envs

        # --- Modification Start ---
        # Convert numpy obs to torch tensors and permute dims for image data
        # Create a NEW dictionary for tensors, leaving obs_seq_input unmodified
        obs_seq_tensor = {}
        model_device = next(self.parameters()).device # Get device from model parameters

        for k, v in obs_seq_input.items(): # Iterate through the input dict
            # Check if it's already a tensor, otherwise convert from numpy
            if isinstance(v, np.ndarray):
                tensor_v = torch.from_numpy(v).to(device=model_device)
            elif isinstance(v, torch.Tensor):
                # If it's already a tensor, just ensure it's on the right device
                tensor_v = v.to(device=model_device)
            else:
                # Handle unexpected types if necessary
                raise TypeError(f"Unsupported observation type for key '{k}': {type(v)}")

            # Permute image data if needed (assuming numpy input: B, T, H, W, C)
            # Ensure the target shape for the network is (B, T, C, H, W)
            if k == 'rgb' and self.include_rgb:
                if tensor_v.ndim != 5 or tensor_v.shape[-1] != 3:
                    raise ValueError(f"Unexpected shape for RGB input: {tensor_v.shape}. Expected (B, T, H, W, 3)")
                obs_seq_tensor[k] = tensor_v.permute(0, 1, 4, 2, 3) # B, T, C, H, W
            elif k == 'depth' and self.include_depth:
                # Handle cases where depth might be (B,T,H,W) or (B,T,H,W,1)
                if tensor_v.ndim == 4: # B, T, H, W -> add channel dim
                    tensor_v = tensor_v.unsqueeze(-1) # B, T, H, W, 1
                if tensor_v.ndim != 5 or tensor_v.shape[-1] != 1:
                    raise ValueError(f"Unexpected shape for Depth input: {tensor_v.shape}. Expected (B, T, H, W, 1)")
                obs_seq_tensor[k] = tensor_v.permute(0, 1, 4, 2, 3) # B, T, C, H, W
            else: # Handle 'state' and other non-image keys (assuming state is B, T, D)
                if k == 'state' and tensor_v.ndim != 3:
                    raise ValueError(f"Unexpected shape for State input: {tensor_v.shape}. Expected (B, T, State_Dim)")
                obs_seq_tensor[k] = tensor_v

        # --- Modification End ---

        # --- Use obs_seq_tensor from now on ---
        obs_cond = self.encode_obs(
            obs_seq_tensor, eval_mode=True # Pass the processed tensor dictionary
        )  # (B, obs_horizon * (visual_feature_dim + obs_state_dim))

        # initialize action from Guassian noise
        noisy_action_seq = torch.randn(
            (B, self.pred_horizon, self.act_dim), device=model_device # Use model_device
        )

        # Set timesteps for the diffusion process
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        # Diffusion sampling loop
        # Use a different loop variable name to avoid conflict if 'k' is used outside
        for t_step in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.noise_pred_net(
                sample=noisy_action_seq,
                # timestep needs to be a tensor matching batch size for the model
                timestep=t_step.unsqueeze(0).repeat(B).to(model_device),
                global_cond=obs_cond,
            )

            # inverse diffusion step (remove noise)
            noisy_action_seq = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t_step, # Use the loop variable t_step directly
                sample=noisy_action_seq,
            ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        action = noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)

        # Return numpy array as expected by env.step
        return action.cpu().numpy()
    
def save_ckpt(run_name, tag, agent_state, ema_agent_state):
    save_dir = f"runs/{run_name}/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "agent": agent_state,
            "ema_agent": ema_agent_state,
        },
        f"{save_dir}/{tag}.pt",
    )
    print(f"Checkpoint saved to {save_dir}/{tag}.pt")

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # Check demo metadata for control mode consistency
    if args.demo_path.endswith(".h5"):
        json_file = args.demo_path[:-2] + "json"
        try:
            with open(json_file, "r") as f:
                demo_info = json.load(f)
                if "env_info" in demo_info and "env_kwargs" in demo_info["env_info"] and "control_mode" in demo_info["env_info"]["env_kwargs"]:
                    control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
                elif "episodes" in demo_info and len(demo_info["episodes"]) > 0 and "control_mode" in demo_info["episodes"][0]:
                    control_mode = demo_info["episodes"][0]["control_mode"]
                else:
                    print(f"Warning: Control mode not found in {json_file}. Cannot verify consistency.")
                    control_mode = args.control_mode # Assume args is correct
                assert (
                    control_mode == args.control_mode
                ), f"Control mode mismatched! Dataset has '{control_mode}', but args specify '{args.control_mode}'."
        except FileNotFoundError:
            print(f"Warning: Metadata file {json_file} not found. Cannot verify control mode consistency.")
        except Exception as e:
             print(f"Warning: Error reading metadata file {json_file}: {e}. Cannot verify control mode consistency.")

    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon, "pred_horizon must be >= obs_horizon + act_horizon - 1"
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Determine video save directory
    video_save_dir = None
    if args.capture_video:
        if args.evaluate and args.checkpoint:
             # Save eval videos in a subfolder next to the checkpoint
             video_save_dir = f"{os.path.dirname(args.checkpoint)}/test_videos_{run_name}"
        else:
             # Save training/default videos in the run directory
             video_save_dir = f"runs/{run_name}/videos"
        print(f"Saving videos to {video_save_dir}")


    # create evaluation environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse", # Eval typically uses sparse rewards
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default") # Optional: Better rendering
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    else:
         # Try to get max_episode_steps from a temporary env if not provided
         try:
             tmp_env = gym.make(args.env_id)
             args.max_episode_steps = tmp_env.spec.max_episode_steps
             env_kwargs["max_episode_steps"] = args.max_episode_steps
             tmp_env.close()
             print(f"Using default max_episode_steps: {args.max_episode_steps}")
         except Exception as e:
             raise ValueError("max_episode_steps must be specified in args or available in the default environment spec.") from e


    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=video_save_dir, # Pass determined video path
        wrappers=[FlattenRGBDObservationWrapper],
        save_trajectory=args.evaluate # Save trajectory only in evaluate mode
    )
    print(f"Evaluation environment created with {args.num_eval_envs} parallel envs.")

    # Agent setup
    agent = Agent(envs, args).to(device)
    ema_agent = Agent(envs, args).to(device) # Agent used for evaluation (holds EMA weights)

    # --- Checkpoint Loading ---
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint_data = torch.load(args.checkpoint, map_location=device)

        # Prioritize loading EMA agent state for evaluation
        if "ema_agent" in checkpoint_data:
            ema_agent.load_state_dict(checkpoint_data["ema_agent"])
            print("Loaded EMA agent state dict.")
            # Also load base agent if available (might be needed if resuming training later)
            if "agent" in checkpoint_data:
                 agent.load_state_dict(checkpoint_data["agent"])
                 print("Loaded base agent state dict.")
        elif "agent" in checkpoint_data:
             # Fallback: If only base agent saved, use it for evaluation
             print("Warning: 'ema_agent' key not found in checkpoint. Using base agent state for evaluation.")
             agent.load_state_dict(checkpoint_data["agent"])
             ema_agent.load_state_dict(checkpoint_data["agent"]) # Copy base agent weights to ema_agent
             print("Loaded base agent state dict and copied to EMA agent.")
        else:
             raise ValueError("Checkpoint file does not contain 'agent' or 'ema_agent' state_dict.")

    # --- Evaluation Mode ---
    if args.evaluate:
        if not args.checkpoint:
            raise ValueError("Checkpoint path (--checkpoint) must be provided for evaluation mode.")

        print("Running in evaluation-only mode.")
        ema_agent.eval() # Set agent to evaluation mode

        eval_metrics = evaluate(
            args.num_eval_episodes,
            ema_agent, # Use the EMA agent for evaluation
            envs,
            device,
            args.sim_backend
            # Removed obs_horizon, pred_horizon, act_horizon
        )

        print(f"\nEvaluation Results ({args.num_eval_episodes} episodes):")
        aggregated_metrics = defaultdict(list)
        for k in eval_metrics.keys(): # eval_metrics is usually a dict of lists
            if isinstance(eval_metrics[k], list) and len(eval_metrics[k]) > 0 :
                mean_val = np.mean(eval_metrics[k])
                std_val = np.std(eval_metrics[k])
                print(f"  {k}: {mean_val:.4f} +/- {std_val:.4f}")
                aggregated_metrics[f"eval/{k}_mean"].append(mean_val)
                aggregated_metrics[f"eval/{k}_std"].append(std_val)
            elif isinstance(eval_metrics[k], (int, float, bool)): # Handle single value metrics if any
                 print(f"  {k}: {eval_metrics[k]}")
                 aggregated_metrics[f"eval/{k}"].append(eval_metrics[k])

        # Optional: Save evaluation metrics to a file
        # === START Replacement: JSON Saving Block ===
        # Save evaluation metrics to a file after converting numpy types
        print("Attempting to save evaluation metrics...")
        serializable_metrics = {}
        for k, v in eval_metrics.items(): # Iterate through the results from evaluate()
            # Primary Case: Value is expected to be a NumPy array from evaluate()
            if isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist() # Convert array to Python list
            # Defensive Case 1: Value is a list containing np scalars (less likely)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.generic):
                serializable_metrics[k] = [item.item() for item in v] # Convert each item
            # Defensive Case 2: Value is a single np scalar
            elif isinstance(v, np.generic):
                serializable_metrics[k] = v.item() # Convert single scalar
            # Case 3: Value is already serializable
            elif isinstance(v, (list, dict, str, int, float, bool, type(None))):
                serializable_metrics[k] = v
            else:
                # Handle unexpected types
                print(f"Warning: Skipping non-serializable type {type(v)} for key '{k}' during JSON save.")
                # Decide how to handle: skip key or store None
                # Option A: Skip key
                # continue
                # Option B: Store None
                serializable_metrics[k] = None

        # Define path (ensure video_save_dir was defined earlier based on args.capture_video)
        # Check if video saving was enabled/path exists before trying to save JSON
        if video_save_dir and os.path.exists(os.path.dirname(video_save_dir)):
            eval_results_path = os.path.join(os.path.dirname(video_save_dir), "eval_results.json")
            try:
                # Save the properly converted dictionary
                with open(eval_results_path, "w") as f:
                    json.dump(serializable_metrics, f, indent=4)
                print(f"Evaluation metrics saved to {eval_results_path}")
            except Exception as e:
                print(f"Warning: Could not save evaluation metrics to JSON: {e}")
        else:
            # Handle case where video directory wasn't created (e.g., capture_video=False)
            print("Skipping saving evaluation metrics (video directory not specified or does not exist).")
        # === END Replacement: JSON Saving Block ===

        print("\nEvaluation finished.")
        envs.close()
        exit() # Exit after evaluation

    # --- Training Mode ---
    print("Running in training mode.")
    if args.track:
        import wandb
        # Format config for wandb
        config = vars(args).copy() # Create a copy to avoid modifying args
        # Add nested dicts for env cfgs if desired
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        # Potentially add demo dataset info too
        # config["demo_dataset_cfg"] = ...
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True, # Sync tensorboard logs automatically
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy", # Group runs by algorithm type
            tags=["diffusion_policy", args.env_id, args.control_mode], # Add relevant tags
        )
        print("Weights & Biases tracking enabled.")
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    print(f"Tensorboard logs will be saved to runs/{run_name}")

    # Dataset and Dataloader setup
    print("Setting up training dataset...")
    # create temporary env to get original observation space as AsyncVectorEnv (CPU parallelization) doesn't permit that
# === Start Replacement Block ===
    import mani_skill.envs # Ensure registration is triggered
    from mani_skill.utils.registration import find_env_spec

    print(f"Fetching spec for {args.env_id} to determine observation space...")
    env_spec = find_env_spec(args.env_id)
    if env_spec is None:
        raise ValueError(f"Could not find environment specification for {args.env_id}")

    # Define env_kwargs needed for space building or temp env (subset of full env_kwargs)
    # Primarily obs_mode and control_mode are usually needed for space definition
    env_kwargs_for_space = dict(
        obs_mode=args.obs_mode,
        control_mode=args.control_mode
        # Add other args.* parameters if the specific env's space depends on them
    )
    # Add max_episode_steps if needed by potential temporary env creation fallback
    if args.max_episode_steps is not None:
        env_kwargs_for_space["max_episode_steps"] = args.max_episode_steps


    original_obs_space = None
    include_rgb = None
    include_depth = None

    # --- Attempt 1: Build space directly from spec (Ideal, but might not exist) ---
    # Check if the spec's class has a method like build_observation_space
    # NOTE: As of ManiSkill 2 (v0.6, v1.0), there isn't a standardized public static method for this.
    # We will proceed directly to the fallback which is more robust across versions.
    # If a future version adds such a method, it could be attempted here first.
    print("Standard ManiSkill spec does not provide static space builder, using fallback.")

    # --- Attempt 2: Fallback - Create temporary env with correct backend ---
    try:
         print(f"Creating temporary env instance with backend: {args.sim_backend} to get space info...")
         # Use args.sim_backend to initialize PhysX correctly if needed, preventing the error
         # Pass only the kwargs needed to determine the space structure
         tmp_env = gym.make(args.env_id, sim_backend=args.sim_backend, **env_kwargs_for_space)
         original_obs_space = tmp_env.observation_space
         # Determine visual data inclusion from the actual temp env's properties
         include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
         include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
         tmp_env.close()
         print("Successfully determined space info from temporary env.")
    except Exception as e_tmp:
         print(f"Error creating temporary env during fallback: {e_tmp}")
         # If fallback fails, we cannot proceed
         raise ValueError("Could not determine observation space using temporary env.") from e_tmp

    # Ensure variables were set
    if original_obs_space is None:
         raise RuntimeError("Failed to determine original_obs_space")

    # Rename variable to match original code if needed (or update dataset call)
    orignal_obs_space = original_obs_space # Keep name consistent if dataset code uses it

    print(f"Determined Observation Space. include_rgb={include_rgb}, include_depth={include_depth}")
    # === End Replacement Block ===

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(
            np.transpose, axes=(0, 3, 1, 2)
        ),  # (B, H, W, C) -> (B, C, H, W)
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth = include_depth # Pass whether depth is included
    )

    dataset = SmallDemoDataset_DiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=orignal_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        device=device, # Preload data to device if memory allows
        num_traj=args.num_demos,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        control_mode=args.control_mode
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    # Use iteration-based sampler to ensure exactly total_iters steps
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )
    print(f"DataLoader created with batch size {args.batch_size} and {args.total_iters} iterations.")

    # Optimizer, Scheduler, EMA setup
    optimizer = optim.AdamW(
        params=agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6
    )

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    # If loaded from checkpoint, ensure EMA internal state matches the loaded ema_agent
    if args.checkpoint and "ema_agent" in checkpoint_data:
        ema.module.load_state_dict(checkpoint_data["ema_agent"])
        print("Initialized EMA internal state from loaded ema_agent checkpoint.")
    elif args.checkpoint and "agent" in checkpoint_data:
        ema.module.load_state_dict(checkpoint_data["agent"]) # Fallback
        print("Initialized EMA internal state from loaded base agent checkpoint (ema_agent not found).")


    # --- Training Loop ---
    best_eval_metrics = defaultdict(lambda: -np.inf) # Initialize with -inf for maximization
    timings = defaultdict(float)
    start_time = time.time()

    agent.train() # Set agent to training mode
    ema_agent.train() # Keep EMA agent in train mode for potential updates? EMA handles this.

    pbar = tqdm(total=args.total_iters, desc="Training", dynamic_ncols=True)
    for iteration, data_batch in enumerate(train_dataloader):
        step_start_time = time.time()
        timings["data_loading"] += step_start_time - start_time # Time since last step start

        # --- Forward Pass ---
        forward_start_time = time.time()
        # Move data to device if not already preloaded
        # observations_batch = {k: v.to(device) for k, v in data_batch['observations'].items()}
        # actions_batch = data_batch['actions'].to(device)
        # Assuming data is preloaded to device in Dataset:
        observations_batch = data_batch['observations']
        actions_batch = data_batch['actions']

        total_loss = agent.compute_loss(
            obs_seq=observations_batch,
            action_seq=actions_batch,
        )
        timings["forward"] += time.time() - forward_start_time

        # --- Backward Pass & Optimization ---
        backward_start_time = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        timings["backward"] += time.time() - backward_start_time

        # --- EMA Update ---
        ema_start_time = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - ema_start_time

        # --- Logging ---
        if iteration % args.log_freq == 0:
            writer.add_scalar("train/loss", total_loss.item(), iteration)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            # Log timings per iteration (optional)
            # for k, v in timings.items():
            #     writer.add_scalar(f"time/{k}_iter", v / (iteration + 1), iteration) # Average time per iter

        # --- Evaluation ---
        if iteration % args.eval_freq == 0 or iteration == args.total_iters - 1:
            eval_start_time = time.time()
            agent.eval() # Set base agent to eval mode for consistency if needed
            ema_agent.eval() # Set EMA agent to eval mode
            # Copy current EMA weights to the evaluation agent
            ema.copy_to(ema_agent.parameters())

            eval_metrics = evaluate(
                args.num_eval_episodes,
                ema_agent, # Evaluate the EMA agent
                envs,
                device,
                args.sim_backend
                # Removed obs_horizon, pred_horizon, act_horizon
            )
            timings["eval"] += time.time() - eval_start_time # Accumulate total eval time

            print(f"\n--- Iteration {iteration} Evaluation ---")
            metrics_updated = False
            for k in eval_metrics.keys(): # eval_metrics is dict of lists
                 if isinstance(eval_metrics[k], list) and len(eval_metrics[k]) > 0:
                    mean_val = np.mean(eval_metrics[k])
                    std_val = np.std(eval_metrics[k])
                    print(f"  eval/{k}: {mean_val:.4f} +/- {std_val:.4f}")
                    writer.add_scalar(f"eval/{k}_mean", mean_val, iteration)
                    writer.add_scalar(f"eval/{k}_std", std_val, iteration)

                    # Check for best metric and save checkpoint
                    save_metric_key = f"{k}_mean" # e.g. success_at_end_mean
                    if mean_val > best_eval_metrics[save_metric_key]:
                         best_eval_metrics[save_metric_key] = mean_val
                         # Prepare states for saving
                         agent_state = agent.state_dict()
                         ema_agent_state = ema_agent.state_dict() # Save the current best EMA state
                         save_ckpt(run_name, f"best_eval_{k}", agent_state, ema_agent_state)
                         metrics_updated = True
                         print(f"    New best {k}: {mean_val:.4f}. Checkpoint saved.")

            agent.train() # Set agents back to train mode
            ema_agent.train()

        # --- Periodic Checkpointing ---
        if args.save_freq is not None and iteration > 0 and iteration % args.save_freq == 0:
             agent_state = agent.state_dict()
             # Get current EMA state for the periodic checkpoint
             current_ema_state = agent.state_dict() # Create placeholder
             ema.copy_to(current_ema_state) # Copy current EMA state
             save_ckpt(run_name, f"iter_{iteration}", agent_state, current_ema_state)


        pbar.update(1)
        pbar.set_postfix(loss=f"{total_loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")
        start_time = time.time() # Reset start time for next iteration's data loading timing

    # --- End of Training ---
    pbar.close()
    print("Training finished.")

    # Save final checkpoint
    agent_state = agent.state_dict()
    final_ema_state = agent.state_dict() # Placeholder
    ema.copy_to(final_ema_state) # Get final EMA state
    save_ckpt(run_name, "final", agent_state, final_ema_state)

    # Log total times
    total_training_time = time.time() - start_time
    writer.add_scalar("time/total_training_seconds", total_training_time, args.total_iters)
    for k, v in timings.items():
        writer.add_scalar(f"time/total_{k}_seconds", v, args.total_iters)
    print(f"Total training time: {total_training_time:.2f} seconds")

    envs.close()
    if writer:
        writer.close()
    if args.track:
        wandb.finish()