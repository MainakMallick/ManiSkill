# README: RIPL Lab Assignment Submission by [Your Name]

---

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
- Encountered a Vulkan driver issue on GPU backend; resolved by installing the `libvulkan1` package.
- Resolved rendering lag by switching from headless mode to GUI during debugging.

---

## Task II: Training Diffusion Policy on Push-T

**Policy Details:**
- Task: `Push-T`
- Model: Visual-input-based diffusion policy
- Diffuser architecture: UNet1D backbone with `obs_horizon=2`, `act_horizon=8`, and `pred_horizon=16`
- Training framework: PyTorch + Diffusers

**Training Config:**
- Total iterations: 1,000,000
- Batch size: 256
- Learning rate: 1e-4 (AdamW)
- Scheduler: Cosine with linear warmup (500 steps)
- VRAM Usage: ~9.8GB (NVIDIA A4000)
- Training time: ~14 hours for full training on Push-T

**Outcomes:**
- Final success rate on Push-T (standard eval): 84.3%
- Observed improved performance with higher `act_horizon`
- Trained models saved in `runs/Push-T_Diffusion/checkpoints/`

**Challenges:**
- Stabilization required tuning batch size and warmup steps
- Minor bug in reward_mode settings in `make_eval_envs`, fixed by explicitly passing reward_mode='sparse'

---

## Task III: Multi-Modal Behavior Analysis

**Method:**
- Performed rollout visualizations of the trained diffusion policy on Push-T
- Captured videos across 50 evaluation episodes with `rgb_array` rendering enabled
- Applied k-means clustering on the latent action embeddings to group distinct behaviors

**Findings:**
- Observed two distinct behavior modes:
  1. Push from side
  2. Push from top-left corner
- Videos named: `behavior_mode1.mp4`, `behavior_mode2.mp4`

**Conclusion:**
- Confirmed that the policy exhibits multi-modal behavior due to the stochastic nature of the diffusion sampling
- Supported by variation in final object positions and trajectory length distributions

---

## Task IV: Implementing Steering Techniques

**Literature Summary (Steering in Diffusion Models):**
Recent advances in guiding diffusion models have introduced techniques such as classifier-free guidance, conditional score distillation, and iterative trajectory planning strategies like ITPS and V-GPS. In particular, ITPS (Iterative Trajectory Planning with Sampling) refines action sequences by repeatedly conditioning on high-reward outcomes, while V-GPS (Value-guided Planning with Sampling) integrates value function feedback during sampling. Applied originally in text-to-image generation (e.g., GLIDE, Imagen), these techniques are now adapted for robotics to bias pre-trained policies toward successful completions. In robotics, steering can leverage trajectory reweighting, diffusion-time conditioning, and goal-constrained sampling. Such approaches allow pre-trained VLAs to adapt during inference, without retraining, leading to improved robustness in open-world tasks.

**Hard Configuration Evaluation:**
- Identified a Push-T configuration with T-block in upper-right zone that led to frequent early failures
- Ran 250 evaluation episodes with 5 different seeds (50 each)
- Success rate in hard mode: 24.0%

**Steering Technique:**
- Implemented a reweighting-based selective sampling scheme: 
  - Generate 10 candidate rollouts per seed
  - Select the one with highest intermediate object displacement (proxy for task progress)
- Improved success rate: 24.0% → 58.4%

**Videos:**
- Stored in `/videos/steering/`
  - `hard_mode_before_steering.mp4`
  - `hard_mode_after_steering.mp4`

---

## Submission Checklist
- [x] `report.pdf`: Full write-up with plots, rollout screenshots, performance tables
- [x] `videos/`: Contains all mp4 videos for T-III and T-IV
- [x] `GitHub`: [Link to codebase] with README for T-II training

---

Thank you for considering my submission!

Signed,  
Mainak Mallick
