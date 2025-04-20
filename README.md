# RIPL Lab Assignment Submission by Mainak Mallick 
---
![Demo](Images/demo.png)
## Task I: ManiSkill Setup

**Environment Setup:**
- Successfully set up the ManiSkill environment by following the [official ManiSkill installation guide](https://github.com/haosulab/ManiSkill). Used Conda for environment isolation.
- Installed all required dependencies using `pip install -e .` within the ManiSkill2 base folder.
- Verified the setup by running:
  - `python -m mani_skill.examples.demo_play` for rendering demo episodes
  - `python -m mani_skill.examples.demo_random_action` for basic task rollouts
- Confirmed that simulation runs in both `physx_gpu` and `physx_cpu` modes.

**Colab Notebook:**
- Implemented each section from the official ManiSkill colab notebook for hands-on familiarity.
- Visualized rollouts and manipulated environment parameters.
- Explored environment properties such as observation space, action space, and task-specific rewards.

**Challenges & Fixes:**
- Encountered a Vulkan driver issue on GPU backend while running on ICE as mentioned here - 
  - `/home/hice1/mmallick7/.local/lib/python3.10/site-packages/sapien/_vulkan_tricks.py:37: UserWarning: Failed to find Vulkan ICD file. This is probably due to an incorrect or partial installation of the NVIDIA driver. SAPIEN will attempt to provide an ICD file anyway but it may not work. warn`. This was unresolved, because I didn't had the permission for running sudo apt-get install libvulkan1 command - though it didn't impact the training process because it's only used for rendering interactive UI.

---

## Task II: Training Diffusion Policy on Push-T

**Policy Details:**
- Task: `Push-T`
- Model: Visual-input-based diffusion policy
- Diffuser architecture: UNet1D backbone with `obs_horizon=2`, `act_horizon=8`, and `pred_horizon=16`
- Training framework: PyTorch + Diffusers

**Training Config:**
- Total iterations: 30,000 (could have gone with 1000,000 iterations but got stuck due to time constrait)
- Batch size: 256
- Learning rate: 1e-4 (AdamW)
- max_episode_steps 100
- Demos - 451
- VRAM Usage: ~12.1GB (NVIDIA L40S)
- Warmup steps - 500
- demos - 451
**Dataset Generation:**

**Training Procedure:** 

**Outcomes:**
- Experimented with prediction horizon and acting horizon combinations-
- Act_horizon, pred_horizon(1): 8,16  
  Act_horizon, pred_horizon(2): 6,12  
  Act_horizon, pred_horizon(3): 10, 18
- Highest success rate on Push-T: 30.4% on 8,16 combination
- Observed near to no performance improvement dufference when the action horizon is modified, but for prediction horizon I abserved maximum succes rate in 8,16 combination and dipped in both ways.
- Trained models saved in `/ManiSkill/scripts/data_generation/runs/diffusion_policy-PushT-v1-rgb-451_motionplanning_demos-1/checkpoints/best_eval_success_at_end.pt`

**Challenges and Comments:**
- Stabilization required tuning batch size and warmup steps.
- As per the issue provided [here](https://github.com/haosulab/ManiSkill/issues/882) there should be 700 demonstartions but I could found only 451 in total.
- Faced challenging situation while creating the trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5 file from the trajectory.none.pd_joint_delta_pos.physx_cuda.h5 file, dense reward was initially not getting genrated,  there was a just a success failure boolean, as a result so there was no improvement while training. Fixed it by explicitly passing reward mode as dense while generating the data.

**Results and Plots:**
Below are snapshots of training metrics visualized using Weights & Biases:

| Chart 1 | Chart 2 | Chart 3 |
|---------|---------|---------|
| ![](Images/WBChart419202561158PM.png) | ![](Images/WBChart419202561207PM.png) | ![](Images/WBChart419202561213PM.png) |
| Chart 4 | Chart 5 | Chart 6 |
| ![](Images/WBChart419202561219PM.png) | ![](Images/WBChart419202561226PM.png) | ![](Images/WBChart419202561240PM.png) |
| Chart 7 | Chart 8        |   Chart 9      |
| ![](Images/WBChart419202561247PM.png) | ![](Images/WBChart419202564221PM.png) | ![](Images/WBChart419202564229PM.png) |


The rest of the plots can be found [here](https://wandb.ai/mainakmallick-georgia-institute-of-technology/ManiSkill/runs/soja5l5v?nw=nwusermainakmallick)
---

## Task III: Multi-Modal Behavior Analysis

**Method:**
- Performed rollout visualizations of the trained diffusion policy on Push-T
- Captured videos across 700 evaluation episodes with `rgb_array` rendering enabled

**Findings:**
- Observed two distinct behavior modes:
  1. Push from Top part
  2. Push from Bottom part
  which leads to a success in demo 1 but failure in demo 2.
- Videos named: [Demo Video 1](videos/demo1.mp4), [Demo Video 2](videos/demo2.mp4)

**Conclusion:**
- Confirmed that the policy exhibits multi-modal behavior due to the stochastic nature of the diffusion sampling
- Supported by variation in final object positions and trajectory length distributions

---

## Task IV: Implementing Steering Techniques
## Literature Review: Steering in Diffusion Model Architectures

Recent advancements in steering diffusion-based generative policies have introduced efficient mechanisms for aligning robot behavior with user intent during inference, without requiring fine-tuning. **Inference-Time Policy Steering (ITPS)** [Wang et al., 2024] proposes guiding a frozen generative policy using real-time human interactions (point goals, trajectory sketches, and physical corrections). It incorporates these forms of input as alignment objectives during the sampling process, either post-hoc (e.g., output perturbation, ranking) or during diffusion (e.g., guided or stochastic sampling). Among these, **Stochastic Sampling (SS)** demonstrated the best trade-off between alignment and constraint satisfaction by approximating sampling from the product of the trajectory and user-intent distributions, instead of their sum, thereby reducing distribution shift and preserving the validity of the generated trajectories. 

In parallel, **Value-Guided Policy Steering (V-GPS)** [Nakamoto et al., 2024] uses a value function trained via offline RL to re-rank actions sampled from generalist policies (e.g., Octo, RT-X, OpenVLA) at test-time. This modular plug-and-play framework improves robustness and precision in real-world robotic manipulation tasks without altering the base policy. Unlike ITPS, which leverages interaction-conditioned objectives for fine-grained trajectory shaping, V-GPS focuses on maximizing task reward alignment through high-value action selection, and has shown consistent improvements across 12 real-world and simulated tasks. 

Together, these works underscore a growing trend in **compute-aware inference-time steering** that enables adaptive, goal-aligned behaviors in pre-trained diffusion or transformer-based policies, and highlight promising directions for scalable, multimodal control in robotics.


**Evaluation script generation:**
After quite a few number of attempts, I couldn't able to find out a standalone evaluation script or configuration of train_rgbd.py which will just run the trained diffusion policy 250 times in this “hard” evaluation episode and report the success rate (SR) in this specific evaluation episode over 5 different random seeds. So I developed one - `test_policy.py`

While developing this script I encountered a couple of challenges - 

**Hard Configuration Evaluation:**
- Identified a Push-T configuration with T-block in upper-right zone that led to frequent early failures in episode 26th which was our "hard" episode.
- Ran 250 evaluation episodes with 5 different seeds (50 each)
- Success rate in hard mode: 24.0%

**Steering Technique:**
Was not able to succesfully implement due to time constraint.

**Videos:**
- Stored in `/videos/steering/`
  - `hard_mode_before_steering.mp4`
  - `hard_mode_after_steering.mp4`

---

## Submission Checklist
- [x] `report.pdf`: Full write-up with plots, rollout screenshots, performance tables
- [x] `videos/`: Contains all mp4 videos for T-III and T-IV
- [x] `GitHub`: [Link to codebase] with README

---

Thank you for considering my submission!

Signed,  
Mainak Mallick
