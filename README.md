# Awesome Diffusion & Flow RL

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![GitHub stars](https://img.shields.io/github/stars/AquaHorseM/Awesome-Diffusion-Flow-RL?style=flat)](https://github.com/AquaHorseM/Awesome-Diffusion-Flow-RL) [![License: CC0-1.0](https://img.shields.io/badge/License-CC0--1.0-blue.svg)](LICENSE)

A curated list of papers and tools for **reinforcement learning and reward optimization with diffusion and flow models** — from image generation and language reasoning to robot control and scientific discovery.

We focus on **algorithms**: how rewards improve a generative policy, how different updates connect, and where to start implementing them. Each paper has a concise, one-sentence summary.

*Last updated: Oct. 24, 2026*

## Overview

### Aim & Scope

This repository offers an algorithm-focused map of reward-driven learning for diffusion and flow models. It covers policy gradients, value-based methods, preference optimization, reward gradients, reward-weighted matching, stochastic control, world-model RL, and closely related inference or distillation methods when they help explain the field.

Method Family and Domain / Task are the two main ways to browse the collection. Rather than cataloging every relevant paper, we curate work that we believe matters to the field and can offer useful ideas or new perspectives to readers exploring it.

> [!NOTE]
> Paper selection and highlights only reflect the maintainer's personal preferences. All short summaries are LLM-generated and human-verified.

- 💡 **Community contributions welcome.** To suggest a paper or correction, please open an [Issue](https://github.com/AquaHorseM/Awesome-Diffusion-Flow-RL/issues) or [Pull Request](https://github.com/AquaHorseM/Awesome-Diffusion-Flow-RL/pulls). If you are interested in helping maintain the repository, please [email us](mailto:qinweimartin@gmail.com). Maintainer access is offered selectively to keep the curation coherent. (Please do not email paper submissions)
- ⭐ **Support this project.** If you find it useful, please cite the repository, give it a star, and share it with the community.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the entry format and contribution guidelines.

<a id="legend"></a>

- ⭐ **Fundamental** — work that helps explain the field.
- 🚀 **Frontier & Potential** — promising current directions.

## Contents

- [Getting Started](#getting-started)
  - [Selected Highlights](#selected-highlights)
  - [Surveys & Reviews](#surveys--reviews)
  - [Tutorials & Starter Resources](#tutorials--starter-resources)
  - [Method Map](#method-map)
- [Core Methods](#core-methods)
  - [Transition-Likelihood Policy Gradients](#transition-likelihood-policy-gradients)
  - [Likelihood-Surrogate Policy Gradients](#likelihood-surrogate-policy-gradients)
  - [Value-Based Methods](#value-based-methods)
  - [Pathwise Reward Gradients](#pathwise-reward-gradients)
  - [Reward-Weighted & Matching-Based Policy Improvement](#reward-weighted--matching-based-policy-improvement)
  - [Other Optimization Methods](#other-optimization-methods)
  - [Inference & Steering](#inference--steering)
  - [Distillation](#distillation)
- [Applications & Domain-Specific Methods](#applications--domain-specific-methods)
  - [Images & Video](#images--video)
  - [World Models & Interactive Video](#world-models--interactive-video)
  - [Language & Reasoning](#language--reasoning)
  - [Robotics & Control](#robotics--control)
  - [Scientific Discovery](#scientific-discovery)
  - [Audio](#audio)
- [Tech Reports](#tech-reports)
- [Further Reading & Resources](#further-reading--resources)
  - [Libraries & Toolkits](#libraries--toolkits)
  - [Background](#background)
  - [Rewards & Feedback](#rewards--feedback)
  - [Theory & Analysis](#theory--analysis)
  - [Related Lists](#related-lists)
- [Citation](#citation)

## Getting Started

### Selected Highlights

> [!IMPORTANT]
> Looking for the papers that best capture the field's foundations and most exciting emerging directions?
> Explore our curated **[Fundamentals and Frontier & Potential highlights](HIGHLIGHTS.md)**.

### Surveys & Reviews

Begin with a ⭐ survey for a broad view, or choose a focused review of control, GRPO, or video alignment.

<a id="survey-2610-21539"></a>

- ⭐ **[Reinforcement Learning-driven Diffusion Models: Language, Vision, and Beyond](https://arxiv.org/abs/2610.21539)**<br>
  Maps reward-driven diffusion post-training across language, vision, robotics, and scientific generation.

<a id="survey-2610-16275"></a>

- ⭐ **[A Survey of Flow Matching in Reinforcement Learning](https://arxiv.org/abs/2610.16275)**<br>
  Explains how flow RL uses rewards and values for policy learning and generative alignment.

<a id="survey-2407-13734"></a>

- ⭐ **[Understanding RL-Based Fine-Tuning of Diffusion Models](https://arxiv.org/abs/2407.13734)**<br>
  Connects policy gradients, reward backpropagation, and reward-weighted fitting through entropy-regularized control.

<a id="survey-2610-00812"></a>

- **[Video Generation Models: A Survey of Post-Training and Alignment](https://arxiv.org/abs/2610.00812)**<br>
  Covers video post-training through supervised adaptation, distillation, reward optimization, and inference guidance.

<a id="survey-2603-06623"></a>

- **[Advances in GRPO for Generation Models: A Survey](https://arxiv.org/abs/2603.06623)**<br>
  Reviews GRPO for generation, with emphasis on stochastic sampling, credit assignment, efficiency, and reward overoptimization.

<a id="survey-2601-06133"></a>

- **[A Review of Online Diffusion Policy RL Algorithms for Scalable Robotic Control](https://arxiv.org/abs/2601.06133)**<br>
  Compares online diffusion-policy updates for robot control, including action gradients, Q weighting, and backpropagation through time.

### Tutorials & Starter Resources

Slides, an author walkthrough, and a closer look at the design choices behind diffusion RL.

<a id="tutorial-t01"></a>

- ⭐ **[Aligning Diffusion Models with Human Preferences — ECCV 2026 Tutorial](https://diffusion-alignment.github.io/)**<br>
  Introduces RL alignment of diffusion and flow models through ECCV 2026 tutorial slides.

<a id="tutorial-t02"></a>

- **[DDPO Author Explanation](https://rl-diffusion.github.io/)**<br>
  Walks through DDPO’s denoising-as-decision-making formulation and reward fine-tuning.

<a id="tutorial-elbo"></a>

- **[Rethinking the Design Space of Diffusion RL / ELBO Design Space](#paper-arxiv-2602-04663)**<br>
  Explains how endpoint likelihood estimators and ELBO choices shape diffusion policy optimization.

<a id="taxonomy"></a>

### Method Map

Find a learning rule, then follow its representative papers across tasks. A diffusion or flow backbone can support several of these updates.

| Method family | Main idea | Start with |
|---|---|---|
| [Transition-likelihood policy gradients](#transition-likelihood-policy-gradients) | Optimize tractable denoising-transition or path likelihoods. | [DDPO](#paper-arxiv-2305-13301), [Flow-GRPO](#paper-arxiv-2505-05470), [DPPO](#paper-arxiv-2409-00588) |
| [Likelihood-surrogate policy gradients](#likelihood-surrogate-policy-gradients) | Use tractable bounds or prediction losses to approximate policy ratios. | [FPO / FMPG](#paper-arxiv-2507-21053), [DiPOD](#paper-arxiv-2606-13795), [Timestep Weighting](#paper-arxiv-2609-32665) |
| [Actor–critic / Q-learning](#value-based-methods) | Improve actions with learned values or advantages. | [QSM](#paper-arxiv-2312-11752), [FQL](#paper-arxiv-2502-02538), [QF3](#paper-arxiv-2610-08789) |
| [Preference optimization](#likelihood-surrogates--elbo-based-optimization) | Learn from comparisons between generated samples. | [Diffusion-DPO](#paper-arxiv-2311-12908), [AbDPO](#paper-arxiv-2403-16576) |
| [Pathwise reward gradients](#pathwise-reward-gradients) / [adjoints](#posterior-matching-stochastic-control--adjoints) | Propagate reward or value derivatives into generation. | [DRaFT](#paper-arxiv-2309-17400), [Adjoint Matching](#paper-arxiv-2409-08861) |
| [Matching-based policy improvement](#reward-weighted--matching-based-policy-improvement) | Fit reward-weighted or reward-corrected denoising targets. | [DiffusionNFT](#paper-arxiv-2509-16117), [RAM](#paper-arxiv-2605-10759), [wd1 / wd1++](#paper-arxiv-2507-08838) |
| [Other optimization methods](#other-optimization-methods) | Approaches including posterior matching, stochastic control, and adjoint methods. | [Adjoint Matching](#paper-arxiv-2409-08861), [RTB](#paper-arxiv-2405-20971), [DDPP](#paper-arxiv-2410-08134) |
| [Inference & steering](#inference--steering) | Guide sampling while keeping generator weights fixed. | [Diffuser](#paper-arxiv-2205-09991), [SVDD](#paper-arxiv-2408-08252), [TIPS](#paper-arxiv-2610-22099) |
| [On-policy distillation](#distillation) | Match a teacher at states visited by the student. | [DiffusionOPD](#paper-arxiv-2605-15055), [dOPSD](#paper-arxiv-2607-04428), [VIDD](#paper-arxiv-2507-00445) |

<a id="cross-domain-methods"></a>

<a id="methods-across-domains"></a>

<a id="general-methods"></a>

## Core Methods

Core learning algorithms organized by how they obtain a policy-improvement update. Placement here highlights the paper's algorithmic contribution rather than claiming universal applicability. **Subdomain** badges record where the original paper actually develops or evaluates the method; combined values share one badge.

<a id="core-rl-denoising-and-stochastic-flow-policies"></a>

<a id="policy-gradients-for-generation"></a>

<a id="policy-gradients--grpo"></a>

<a id="transition-likelihood-policy-gradients"></a>

### Transition-Likelihood Policy Gradients

<a id="paper-arxiv-2502-01384"></a>

- ⭐ **[SEPO — Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods](https://arxiv.org/abs/2502.01384)** (2025) · [Code (partial release)](https://github.com/ozekri/SEPO) · ![Subdomain: Language & Reasoning · Scientific Discovery](https://img.shields.io/badge/Subdomain-Language%20%26%20Reasoning%20%C2%B7%20Scientific%20Discovery-ebe3e3.svg?style=flat&labelColor=e8ecef)<br>
  Derives policy gradients through concrete scores for discrete diffusion models.

<a id="paper-arxiv-2409-00588"></a>

- ⭐ **[DPPO — Diffusion Policy Policy Optimization](https://arxiv.org/abs/2409.00588)** (2024) · [Code](https://github.com/irom-princeton/dppo) · [Project](https://diffusion-ppo.github.io/) · *online* · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Optimizes action-chunk denoising inside an outer environment MDP with PPO-style credit assignment.

<a id="paper-arxiv-2305-13301"></a>

- ⭐ **[DDPO — Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301)** (2023) · [Code](https://github.com/jannerm/ddpo) · [Project](https://rl-diffusion.github.io/) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Treats denoising transitions as policy actions and optimizes terminal rewards with score-function and PPO-style updates.

<a id="paper-arxiv-2605-03065"></a>

- 🚀 **[OGPO — Sample Efficient Full-Finetuning of Generative Control Policies](https://arxiv.org/abs/2605.03065)** (2026) · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Uses replay-trained critics as terminal rewards for PPO-style improvement through the full generative policy.

<a id="paper-arxiv-2505-22094"></a>

- 🚀 **[ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning](https://arxiv.org/abs/2505.22094)** (2025) · [Code](https://github.com/ReinFlow/ReinFlow) · [Project](https://reinflow.github.io/) · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Injects learned stochasticity into a flow policy to support online policy-gradient fine-tuning for control.

<a id="paper-arxiv-2505-05470"></a>

- 🚀 **[Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)** (2025) · [Code](https://github.com/yifan123/flow_grpo) · [Project](https://gongyeliu.github.io/Flow-GRPO/) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Stochasticizes a flow sampler to obtain scored transitions for group-relative reward optimization.

<a id="paper-arxiv-2606-11025"></a>

- **[Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models](https://arxiv.org/abs/2606.11025)** (2026) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Uses a Gaussian-transition divergence mask to regulate updates instead of relying solely on sampled ratio clipping.

<a id="paper-arxiv-2606-06967"></a>

- **[GenPO++: Generative Policy Optimization with Jacobian-free Likelihood Ratios](https://arxiv.org/abs/2606.06967)** (2026) · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Uses a reversible history-state solver for Jacobian-free likelihood-ratio updates.

<a id="paper-arxiv-2305-16381"></a>

- **[DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2305.16381)** (2023) · [Code](https://github.com/google-research/google-research/tree/master/dpok) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Combines image reward policy gradients with reference-distribution regularization.

<a id="endpoint--elbo-policy-optimization"></a>

<a id="likelihood-surrogates--timestep-weighting"></a>

<a id="likelihood-surrogates--elbo-based-optimization"></a>

<a id="likelihood-surrogate-policy-gradients"></a>

### Likelihood-Surrogate Policy Gradients

<a id="paper-arxiv-2507-21053"></a>

- ⭐ **[FPO / FMPG — Flow Matching Policy Gradients](https://arxiv.org/abs/2507.21053)** (2025) · [Code](https://github.com/akanazawa/fpo) · [Project](https://flowreinforce.github.io/) · ![Subdomain: Images & Video · Robotics & Control](https://img.shields.io/badge/Subdomain-Images%20%26%20Video%20%C2%B7%20Robotics%20%26%20Control-dfe9e6.svg?style=flat&labelColor=e8ecef)<br>
  Builds a PPO-style ratio proxy from flow-matching losses, decoupling action sampling from a fixed denoising chain.

<a id="paper-arxiv-2311-12908"></a>

- ⭐ **[Diffusion-DPO — Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908)** (2023) · [Code](https://github.com/SalesforceAIResearch/DiffusionDPO) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Fits pairwise preferences through a variational diffusion-likelihood objective.

<a id="paper-arxiv-2609-32665"></a>

- 🚀 **[Timestep Weighting: A Hidden Key to Effective ELBO-Based Flow-Matching RL](https://arxiv.org/abs/2609.32665)** (2026) · ![Subdomain: Images & Video · Robotics & Control](https://img.shields.io/badge/Subdomain-Images%20%26%20Video%20%C2%B7%20Robotics%20%26%20Control-dfe9e6.svg?style=flat&labelColor=e8ecef)<br>
  Studies task- and stage-dependent training-noise allocation for ELBO-based flow policy optimization.
  <br><sub>Evaluated on images and robot control; no DLM experiments.</sub>

<a id="paper-arxiv-2606-13795"></a>

- 🚀 **[DiPOD: Diffusion Policy Optimization without Drifting Apart](https://arxiv.org/abs/2606.13795)** (2026) · [Code](https://github.com/Astro-Eric/DiPOD-release) · [Project](https://astro-eric.github.io/blogs/dipod/) · ![Subdomain: Language & Reasoning · Robotics & Control](https://img.shields.io/badge/Subdomain-Language%20%26%20Reasoning%20%C2%B7%20Robotics%20%26%20Control-e3e5e5.svg?style=flat&labelColor=e8ecef)<br>
  Adds on-policy ELBO regularization to variational policy optimization; evaluates language and robot control.
  <br><sub>Uses initial distillation for robot control and repeated regularization for language.</sub>

<a id="paper-arxiv-2602-04663"></a>

- 🚀 **[Rethinking Diffusion RL / EPG — Rethinking the Design Space of Reinforcement Learning for Diffusion Models: On the Importance of Likelihood Estimation Beyond Loss Design](https://arxiv.org/abs/2602.04663)** (2026) · [Code](https://github.com/jaemoo-choi/rethinking-diffusion-rl) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Separates endpoint-likelihood estimator design from the outer policy-gradient objective.

<a id="paper-arxiv-2602-02481"></a>

- 🚀 **[FPO++ — Flow Policy Gradients for Robot Control](https://arxiv.org/abs/2602.02481)** (2026) · [Code](https://github.com/amazon-far/fpo-control) · [Project](https://hongsukchoi.github.io/fpo-control) · *online* · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Refines flow-policy optimization for robot control with per-noise-sample proxies and asymmetric trust regions.

<a id="paper-arxiv-2510-09541"></a>

- 🚀 **[SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models](https://arxiv.org/abs/2510.09541)** (2025) · [Code](https://github.com/facebookresearch/SPG) · [Project](https://chenyuwang-monica.github.io/spg/) · ![Subdomain: Language & Reasoning](https://img.shields.io/badge/Subdomain-Language%20%26%20Reasoning-e5e0f1.svg?style=flat&labelColor=e8ecef)<br>
  Uses upper and lower likelihood bounds to stabilize diffusion-language policy updates.

<a id="paper-arxiv-2606-05468"></a>

- **[FlowPRO / RPRO — FlowPRO: Reward-Free Reinforced Fine-Tuning of Flow-Matching VLAs via Proximalized Preference Optimization](https://arxiv.org/abs/2606.05468)** (2026) · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Anchors flow-VLA implicit rewards with proximal preference optimization on paired failure/correction trajectories, as used in [HyVLA-0.5](#paper-arxiv-2606-14409).

<a id="paper-arxiv-2603-06743"></a>

- **[StableDRL — Stabilizing Reinforcement Learning for Diffusion Language Models](https://arxiv.org/abs/2603.06743)** (2026) · [Code](https://github.com/JianyuanZhong/StableDRL) · ![Subdomain: Language & Reasoning](https://img.shields.io/badge/Subdomain-Language%20%26%20Reasoning-e5e0f1.svg?style=flat&labelColor=e8ecef)<br>
  Uses unconditional clipping and self-normalization to stabilize GRPO updates from estimated diffusion-language likelihood ratios.

<a id="paper-arxiv-2505-19223"></a>

- **[LLaDA 1.5 / VRPO — LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models](https://arxiv.org/abs/2505.19223)** (2025) · [Code](https://github.com/ML-GSAI/LLaDA-1.5) · [Project](https://ml-gsai.github.io/LLaDA-1.5-Demo/) · ![Subdomain: Language & Reasoning](https://img.shields.io/badge/Subdomain-Language%20%26%20Reasoning-e5e0f1.svg?style=flat&labelColor=e8ecef)<br>
  Controls variance in variational preference alignment for a masked language model.

<a id="core-rl-actorcritic--q--advantage"></a>

<a id="value-based-policy-improvement"></a>

### Value-Based Methods

<a id="paper-arxiv-2502-02538"></a>

- ⭐ **[FQL — Flow Q-Learning](https://arxiv.org/abs/2502.02538)** (2025) · [Code](https://github.com/seohongpark/fql) · [Project](https://seohong.me/projects/fql/) · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Learns a behavior flow and a Q-improved one-step actor through distillation.

<a id="paper-arxiv-2502-02316"></a>

- ⭐ **[DIME:Diffusion-Based Maximum Entropy Reinforcement Learning](https://arxiv.org/abs/2502.02316)** (2025) · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Builds maximum-entropy diffusion policy iteration from an entropy-bound Bellman construction.

<a id="paper-arxiv-2312-11752"></a>

- ⭐ **[QSM — Learning a Diffusion Model Policy from Rewards via Q-Score Matching](https://arxiv.org/abs/2312.11752)** (2023) · [Code](https://github.com/escontra/score_matching_rl) · [Project](https://www.michaelpsenka.io/qsm/) · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Connects diffusion policy-score improvement with action gradients of a learned Q-function.

<a id="paper-arxiv-2208-06193"></a>

- ⭐ **[Diffusion-QL — Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/abs/2208.06193)** (2022) · [Code](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Combines a diffusion behavior constraint with Q-maximizing policy updates.

<a id="paper-arxiv-2601-14234"></a>

- 🚀 **[QAM — Q-learning with Adjoint Matching](https://arxiv.org/abs/2601.14234)** (2026) · [Code](https://github.com/ColinQiyangLi/qam) · *online / offline* · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Couples TD learning to adjoint-based improvement of the full iterative diffusion or flow actor.

<a id="paper-arxiv-2610-08789"></a>

- **[QF3: Fast Flow RL with Filtered Q-Gradients](https://arxiv.org/abs/2610.08789)** (2026) · *online* · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Combines flow matching with filtered critic gradients through a one-step output estimate.

<a id="paper-arxiv-2304-10573"></a>

- **[IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies](https://arxiv.org/abs/2304.10573)** (2023) · [Code](https://github.com/philippe-eecs/IDQL) · *offline* · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Fits an IQL critic and extracts actions from a supervised diffusion behavior model.

<a id="reward-gradients"></a>

<a id="pathwise-reward-gradients"></a>

### Pathwise Reward Gradients

<a id="paper-arxiv-2309-17400"></a>

- ⭐ **[DRaFT — Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400)** (2023) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Differentiates a reward through full or truncated denoising trajectories.

<a id="paper-arxiv-2509-06942"></a>

- **[Direct-Align / SRPO — Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference](https://arxiv.org/abs/2509.06942)** (2025) · [Code](https://github.com/Tencent-Hunyuan/SRPO) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Fine-tunes the generator across diffusion timesteps using single-step image reconstruction and fine-grained, text-conditioned relative preference rewards.

<a id="paper-arxiv-2410-13643"></a>

- **[DRAKES — Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design](https://arxiv.org/abs/2410.13643)** (2024) · [Code](https://github.com/ChenyuWang-Monica/DRAKES) · ![Subdomain: Language & Reasoning · Scientific Discovery](https://img.shields.io/badge/Subdomain-Language%20%26%20Reasoning%20%C2%B7%20Scientific%20Discovery-ebe3e3.svg?style=flat&labelColor=e8ecef)<br>
  Backpropagates through relaxed discrete denoising for reward-driven biological sequence design.

<a id="reward-weighted-and-forward-matching"></a>

<a id="reward-weighted--forward-matching"></a>

<a id="reward-weighted--matching-based-policy-improvement"></a>

### Reward-Weighted & Matching-Based Policy Improvement

<a id="paper-arxiv-2509-25050"></a>

- ⭐ **[AWM — Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models](https://arxiv.org/abs/2509.25050)** (2025) · [Code](https://github.com/scxue/advantage_weighted_matching) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Derives advantage-weighted score/flow matching and practical stabilized objectives.

<a id="paper-arxiv-2605-10759"></a>

- 🚀 **[RAM — Reinforce Adjoint Matching: Scaling RL Post-Training of Diffusion and Flow-Matching Models](https://arxiv.org/abs/2605.10759)** (2026) · [Code](https://github.com/AndreasBergmeister/ram) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Constructs reward-corrected detached matching targets for black-box reward optimization.

<a id="paper-arxiv-2509-16117"></a>

- 🚀 **[DiffusionNFT: Online Diffusion Reinforcement with Forward Process](https://arxiv.org/abs/2509.16117)** (2025) · [Code](https://github.com/NVlabs/DiffusionNFT) · [Project](https://research.nvidia.com/labs/cosmos-lab/diffusionnft/) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Uses signed, advantage-guided forward-process fitting with a moving reference, avoiding retained reverse trajectories.

<a id="paper-arxiv-2507-08838"></a>

- 🚀 **[wd1 / wd1++ — wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](https://arxiv.org/abs/2507.08838)** (2025) · [Code](https://github.com/xiaohangt/wd1) · ![Subdomain: Language & Reasoning](https://img.shields.io/badge/Subdomain-Language%20%26%20Reasoning-e5e0f1.svg?style=flat&labelColor=e8ecef)<br>
  Uses weighted log-likelihood and negative-sample unlearning rather than a PPO ratio.

<a id="paper-arxiv-2512-21829"></a>

- **[Tilt Matching for Scalable Sampling and Fine-Tuning](https://arxiv.org/abs/2512.21829)** (2025) · ![Subdomain: Images & Video · Scientific Discovery](https://img.shields.io/badge/Subdomain-Images%20%26%20Video%20%C2%B7%20Scientific%20Discovery-e6e7e4.svg?style=flat&labelColor=e8ecef)<br>
  Uses incremental reward tilting and covariance-based regression to improve generative distributions.

<a id="posterior-matching-stochastic-control-and-adjoints"></a>

<a id="posterior-matching--stochastic-control"></a>

<a id="posterior-matching-stochastic-control--adjoints"></a>

### Other Optimization Methods

<a id="paper-arxiv-2409-08861"></a>

- ⭐ **[Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control](https://arxiv.org/abs/2409.08861)** (2024) · [Code](https://github.com/microsoft/soc-fine-tuning-sd) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Turns stochastic-control fine-tuning into lean-adjoint regression with a memoryless noise construction.

<a id="paper-arxiv-2602-07132"></a>

- 🚀 **[Discrete Adjoint Matching](https://arxiv.org/abs/2602.07132)** (2026) · ![Subdomain: Language & Reasoning](https://img.shields.io/badge/Subdomain-Language%20%26%20Reasoning-e5e0f1.svg?style=flat&labelColor=e8ecef)<br>
  Extends adjoint-style reward optimization to discrete CTMC generators through multiplicative rate corrections.

<a id="paper-arxiv-2608-14430"></a>

- **[Designing Reinforcement Learning for Diffusion Models: A Unified Path-Space View](https://arxiv.org/abs/2608.14430)** (2026) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Relates reward optimization estimators through path importance sampling and value-gradient corrections.

<a id="paper-arxiv-2410-08134"></a>

- **[DDPP — Steering Masked Discrete Diffusion Models via Discrete Denoising Posterior Prediction](https://arxiv.org/abs/2410.08134)** (2024) · ![Subdomain: Images & Video · Language & Reasoning · Scientific Discovery](https://img.shields.io/badge/Subdomain-Images%20%26%20Video%20%C2%B7%20Language%20%26%20Reasoning%20%C2%B7%20Scientific%20Discovery-e6e5e8.svg?style=flat&labelColor=e8ecef)<br>
  Learns reward-tilted discrete denoising posteriors with endpoint-prediction objectives.

<a id="paper-arxiv-2405-20971"></a>

- **[RTB — Amortizing intractable inference in diffusion models for vision, language, and control](https://arxiv.org/abs/2405.20971)** (2024) · [Code](https://github.com/GFNOrg/diffusion-finetuning) · ![Subdomain: Images & Video · Language & Reasoning · Robotics & Control](https://img.shields.io/badge/Subdomain-Images%20%26%20Video%20%C2%B7%20Language%20%26%20Reasoning%20%C2%B7%20Robotics%20%26%20Control-e1e6e9.svg?style=flat&labelColor=e8ecef)<br>
  Trains reward-tilted samplers through relative trajectory-balance residuals and a learned normalizer.

<a id="adjacent-inference-control"></a>

### Inference & Steering

Guide a fixed generator with values, resampling, or a learned controller. A steering policy can learn from reward while the base generator stays frozen.

<a id="guidance-selection-and-learned-steering"></a>

<a id="paper-arxiv-2205-09991"></a>

- ⭐ **[Diffuser — Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/abs/2205.09991)** (2022) · [Code](https://github.com/jannerm/diffuser) · [Project](https://diffusion-planning.github.io/)<br>
  Plans trajectories by denoising with reward and value guidance at inference.

<a id="paper-arxiv-2506-15799"></a>

- 🚀 **[DSRL — Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799)** (2025) · [Code](https://github.com/ajwagen/dsrl) · [Project](https://diffusion-steering.github.io/)<br>
  Learns a latent/noise steering policy around a frozen diffusion or flow action generator.

<a id="paper-arxiv-2501-06848"></a>

- **[Feynman–Kac steering — A General Framework for Inference-time Scaling and Steering of Diffusion Models](https://arxiv.org/abs/2501.06848)** (2025) · [Code](https://github.com/zacharyhorvitz/Fk-Diffusion-Steering)<br>
  Uses particle potentials and resampling to steer generation with fixed model weights.

<a id="paper-arxiv-2408-08252"></a>

- **[SVDD — Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding](https://arxiv.org/abs/2408.08252)** (2024) · [Code](https://github.com/masa-ue/SVDD)<br>
  Selects denoising candidates using soft future values, without reward-updating the generator.

**See also:** [TIPS](#paper-arxiv-2610-22099) — RL-trained DLM visibility controller; [IDQL](#paper-arxiv-2304-10573) — value-based implicit action extraction.

<a id="adjacent-opd--distillation"></a>

### Distillation

Train a student on teacher predictions, often at states the student visits. A reward-trained teacher and its distilled student use different learning objectives.

<a id="teacher--student-optimization"></a>

<a id="paper-arxiv-2511-13649"></a>

- 🚀 **[DMDR — Distribution Matching Distillation Meets Reinforcement Learning](https://arxiv.org/abs/2511.13649)** (2025) · [Code (ImageNet demo)](https://github.com/vvvvvjdy/dmdr)<br>
  Starts with reward-tilted distribution matching, then jointly optimizes distillation and reward objectives so preference alignment guides few-step compression while distillation limits reward hacking.

<a id="paper-arxiv-2607-04428"></a>

- **[dOPSD: On-Policy Self-Distillation for Diffusion Language Models](https://arxiv.org/abs/2607.04428)** (2026)<br>
  Self-distills masked-token predictions from later, more-decoded states of the same rollout, using the shared model as a better-informed teacher.

<a id="paper-arxiv-2605-15055"></a>

- **[DiffusionOPD: A Unified Perspective of On-Policy Distillation in Diffusion Models](https://arxiv.org/abs/2605.15055)** (2026) · [Code](https://github.com/ali-vilab/DiffusionOPD) · [Project](https://quanhaol.github.io/DiffusionOPD-site/)<br>
  Matches teacher transitions on student-visited states with analytic Gaussian reverse-KL / mean targets.

<a id="paper-arxiv-2605-08063"></a>

- **[Flow-OPD: On-Policy Distillation for Flow Matching Models](https://arxiv.org/abs/2605.08063)** (2026) · [Code](https://github.com/CostaliyA/Flow-OPD) · [Project](https://costaliya.github.io/Flow-OPD/)<br>
  Distills reward-specialized flow teachers on student rollouts with task routing and a task-agnostic teacher that anchors the generation manifold.

<a id="paper-arxiv-2507-00445"></a>

- **[VIDD — Iterative Distillation for Reward-Guided Fine-Tuning of Diffusion Models in Biomolecular Design](https://arxiv.org/abs/2507.00445)** (2025) · [Code](https://github.com/divelab/VIDD) · [Project](https://divelab.github.io/VIDD/)<br>
  Distills simulated soft-optimal policies into a diffusion generator using value-informed supervision.

<a id="applications--domain-specific-methods"></a>

## Applications & Domain-Specific Methods

Task- and modality-specific formulations, rewards, architectures, and evaluation settings. Reusable learning algorithms appear in [Core Methods](#core-methods).

<a id="vision--image--video"></a>

### Images & Video

Image- and video-specific preference scorers, temporal objectives, and alignment frameworks.

<a id="broader-training-time-reward-optimization"></a>

<a id="paper-arxiv-2304-05977"></a>

- ⭐ **[ImageReward / ReFL — ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977)** (2023) · [Code](https://github.com/zai-org/ImageReward)<br>
  Pairs a human-preference image scorer with ReFL, which backpropagates reward through late denoising steps.

<a id="paper-arxiv-2505-07818"></a>

- 🚀 **[DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818)** (2025) · [Code](https://github.com/XueZeyue/DanceGRPO) · [Project](https://dancegrpo.github.io/)<br>
  Applies a common group-relative reward recipe across image and video generators.

<a id="paper-arxiv-2501-13918"></a>

- **[VideoAlign / VideoReward — Improving Video Generation with Human Feedback](https://arxiv.org/abs/2501.13918)** (2025) · [Code](https://github.com/KlingAIResearch/VideoAlign) · [Project](https://gongyeliu.github.io/videoalign/)<br>
  Separates flow-DPO and reward-weighted training from noisy reward guidance at inference.

<a id="paper-arxiv-2407-08737"></a>

- **[VADER / Video reward gradients — Video Diffusion Alignment via Reward Gradients](https://arxiv.org/abs/2407.08737)** (2024) · [Code](https://github.com/mihirp1998/VADER) · [Project](https://vader-vid.github.io/)<br>
  Extends differentiable reward alignment to video, including temporal objectives.

**See also:** [DDPO](#paper-arxiv-2305-13301) and [Flow-GRPO](#paper-arxiv-2505-05470) — denoising policy gradients; [Diffusion-DPO](#paper-arxiv-2311-12908) — pairwise preferences; [DRaFT](#paper-arxiv-2309-17400) — reward backpropagation; [forward matching methods](#reward-weighted--forward-matching).

### World Models & Interactive Video

Reward post-training for diffusion and flow world generators, including interactive video and robot rollouts.

<a id="reward-updated-world-generators"></a>

<a id="paper-arxiv-2603-25685"></a>

- 🚀 **[PersistWorld — Persistent Robot World Models: Stabilizing Multi-Step Rollouts via Reinforcement Learning](https://arxiv.org/abs/2603.25685)** (2026) · [Code](https://github.com/Jai2500/PersistWorld) · [Project](https://jaibardhan.com/persistworld/)<br>
  Fits reward-contrasted denoising targets on self-generated robot rollouts to improve persistent, multi-view world models.

<a id="paper-arxiv-2602-09022"></a>

- 🚀 **[WorldCompass: Reinforcement Learning for Long-Horizon World Models](https://arxiv.org/abs/2602.09022)** (2026) · [Code](https://github.com/Tencent-Hunyuan/HY-WorldPlay/tree/main/worldcompass) · [Project](https://3d-models.hunyuan.tencent.com/world/)<br>
  Shares generated video prefixes and ranks candidate clips by reward to train interactive flow world models with negative-aware matching.

<a id="paper-arxiv-2608-07981"></a>

- **[PhyS / Temporal Credit Routing — Distilling Physical Priors into Streaming World Models](https://arxiv.org/abs/2608.07981)** (2026) · [Project](https://lyongo.github.io/PhyS/)<br>
  Assigns physical rewards from overlapping video windows to the corresponding denoising actions in streaming video RL.
  <br><sub>Evaluates streaming physics rather than keyboard-action control; code and data were unreleased at the cutoff.</sub>

<a id="paper-arxiv-2608-04964"></a>

- **[WorldCycle: Self-Verifiable Reinforcement Learning for Long-Horizon Video World Models](https://arxiv.org/abs/2608.04964)** (2026) · [Project](https://nevsnev.github.io/Worldcycle/)<br>
  Turns reversible action cycles into spatial-closure and repeated-cycle rewards for long-horizon world-model post-training.
  <br><sub>Training code was announced but unreleased at the cutoff.</sub>

<a id="paper-arxiv-2603-17808"></a>

- **[EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards](https://arxiv.org/abs/2603.17808)** (2026) · [Code](https://github.com/RobbinW/EVA) · [Project](https://eva-project-page.github.io/)<br>
  Rewards smooth, motion-limit-compliant actions decoded by inverse dynamics so robot video generators improve action executability.

<a id="paper-arxiv-2512-01952"></a>

- **[GrndCtrl / RLWG — GrndCtrl: Grounding World Models via Self-Supervised Reward Alignment](https://arxiv.org/abs/2512.01952)** (2025) · [Project](https://rlwg-grndctrl.github.io/)<br>
  Uses geometric and perceptual self-supervision to align action-conditioned video rollouts with commanded camera motion.
  <br><sub>The project repository was a release placeholder at the cutoff.</sub>

<a id="paper-arxiv-2509-23958"></a>

- **[RLIR — Reinforcement Learning with Inverse Rewards for World Model Post-training](https://arxiv.org/abs/2509.23958)** (2025)<br>
  Uses inverse dynamics to recover commands from generated video and reward-update the diffusion/flow branch for action following.
  <br><sub>This entry covers the diffusion/flow branch; MineWorld uses an autoregressive model.</sub>

**See also:** [Flow-GRPO](#paper-arxiv-2505-05470) — stochastic-flow policy gradients; [DiffusionNFT](#paper-arxiv-2509-16117) — forward-process reward matching; [VideoAlign / VideoReward](#paper-arxiv-2501-13918) — multidimensional video feedback.

<a id="language--dlm"></a>

### Language & Reasoning

Reward training for diffusion language models (DLMs), covering token generation, reasoning, and learned unmasking decisions.

<a id="policy-optimization-and-learned-controllers"></a>

<a id="paper-arxiv-2504-12216"></a>

- 🚀 **[d1 / diffu-GRPO — d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2504.12216)** (2025) · [Code](https://github.com/dllm-reasoning/d1) · [Project](https://dllm-reasoning.github.io/)<br>
  Adapts group-relative reward post-training to masked language models using sequence-likelihood surrogates.

<a id="paper-arxiv-2610-22099"></a>

- **[TIPS: A Tiny Inference-time Policy for Scaling in Diffusion Language Models](https://arxiv.org/abs/2610.22099)** (2026)<br>
  Trains a small Bernoulli unmask/remask controller with GRPO around a frozen language generator.

<a id="paper-arxiv-2603-13319"></a>

- **[LightningRL: Breaking the Accuracy-Parallelism Trade-off of Block-wise dLLMs via Reinforcement Learning](https://arxiv.org/abs/2603.13319)** (2026) · [Code](https://github.com/SJTU-DENG-Lab/LightningRL)<br>
  Combines correctness and parallelism rewards with NLL anchoring for block-wise DLM policy optimization.

<a id="paper-arxiv-2512-09106"></a>

- **[Learning Unmasking Policies for Diffusion Language Models](https://arxiv.org/abs/2512.09106)** (2025) · [Code](https://github.com/apple-aiml-research/ml-rl-dllm)<br>
  Learns reward-driven unmasking decisions, separating position control from token generation.

<a id="paper-arxiv-2509-01142"></a>

- **[Dream-Coder 7B: An Open Diffusion Language Model for Code](https://arxiv.org/abs/2509.01142)** (2025)<br>
  Combines code diffusion pretraining and supervised fine-tuning with unit-test GRPO, complementary masking, and asymmetric clipping.

<a id="paper-arxiv-2506-20639"></a>

- **[DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation](https://arxiv.org/abs/2506.20639)** (2025) · [Code](https://github.com/apple-aiml-research/ml-diffucoder) · [Project](https://machinelearning.apple.com/research/diffucoder)<br>
  Uses complementary masks in coupled-GRPO to reduce token log-likelihood estimation variance during code-model reward training.

<a id="broader-training-time-reward-optimization-1"></a>

<a id="preferences--reward-weighted-training"></a>

**See also:** [SEPO](#paper-arxiv-2502-01384) — discrete policy gradients; [likelihood-surrogate methods](#likelihood-surrogates--elbo-based-optimization) — SPG, StableDRL, VRPO, and DiPOD; [wd1 / wd1++](#paper-arxiv-2507-08838) — ratio-free weighting; [Discrete Adjoint Matching](#paper-arxiv-2602-07132) — CTMC optimization.

<a id="robotics--control--vla"></a>

### Robotics & Control

Control-specific representations and vision-language-action (VLA) frameworks.

<a id="core-rl-generative-policy-optimization"></a>

#### One-step policy optimization

<a id="paper-arxiv-2602-13810"></a>

- 🚀 **[MVP — Mean Flow Policy with Instantaneous Velocity Constraint for One-step Action Generation](https://arxiv.org/abs/2602.13810)** (2026) · [ICLR 2026 Oral](https://iclr.cc/virtual/2026/poster/10007588)<br>
  Combines an instantaneous-velocity-constrained MeanFlow policy with TD learning and Q-ranked action selection for one-step offline-to-online robot manipulation.

<a id="paper-arxiv-2601-20701"></a>

- **[One-Step OGPO — One-Step Generative Policy Optimization for Real-Time Robot Control](https://arxiv.org/abs/2601.20701)** (2026)<br>
  Uses scored stochastic one-step transitions and PPO for real-time control; earlier versions used the DMPO title.

<a id="paper-arxiv-2508-13904"></a>

- **[OFQL — One-Step Flow Q-Learning: Addressing the Diffusion Policy Bottleneck in Offline Reinforcement Learning](https://arxiv.org/abs/2508.13904)** (2025) · [Project](https://thanhkaist.github.io/ofqlpage/) · [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/14f75513f0f1ca01de1e826b52e6b840-Abstract-Conference.html)<br>
  Directly Q-optimizes an average-velocity policy that generates actions in one step during both offline training and inference.

#### Multi-agent coordination

<a id="paper-arxiv-2610-01882"></a>

- **[OMAF — Flowing Faster to Coordinate: One-Step Online Multi-Agent Flow Policies](https://arxiv.org/abs/2610.01882)** (2026)<br>
  Replaces iterative diffusion sampling with a one-step Transformer flow policy and jointly optimizes a path-score surrogate with softmax Q-value estimation for online MARL.

<a id="paper-arxiv-2602-18291"></a>

- **[OMAD — Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies](https://arxiv.org/abs/2602.18291)** (2026) · [Code](https://github.com/lizr16/OMAD) · [ICML](https://proceedings.mlr.press/v306/li26jv.html)<br>
  Trains decentralized diffusion actors with a joint distributional critic and a scaled joint-entropy surrogate for coordinated online off-policy MARL.

<a id="paper-arxiv-2305-17330"></a>

- **[MADiff — Offline Multi-Agent Learning with Diffusion Models](https://arxiv.org/abs/2305.17330)** (2023) · [Code](https://github.com/zbzhu99/madiff) · [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/07e278a120830b10aae20cc600a8c07b-Abstract-Conference.html)<br>
  Uses attention-based trajectory diffusion to model cross-agent coordination, supporting decentralized policies with teammate modeling and centralized control in offline MARL.

<a id="core-rl-vla-policy-improvement"></a>

#### Vision-language-action policies

<a id="paper-arxiv-2511-14759"></a>

- **[RECAP / pi-star0.6 — $\pi^{*}_{0.6}$: a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759)** (2025) · [Project](https://www.pi.website/blog/pistar06)<br>
  Learns advantage-conditioned VLA policies from demonstrations, autonomous rollouts, and expert interventions, using learned values to extract improved behavior.

<a id="paper-arxiv-2510-25889"></a>

- **[piRL — $\pi_\texttt{RL}$: Online RL Fine-tuning for Flow-based Vision-Language-Action Models](https://arxiv.org/abs/2510.25889)** (2025) · [Code](https://github.com/RLinf/RLinf)<br>
  Fine-tunes flow VLAs with Flow-Noise learnable stochastic transitions or Flow-SDE updates over a two-layer denoising/environment MDP.

**See also:** [value-based methods](#value-based-methods) — including [FQL](#paper-arxiv-2502-02538) and [DIME](#paper-arxiv-2502-02316); [DPPO](#paper-arxiv-2409-00588) and [ReinFlow](#paper-arxiv-2505-22094) — action-policy gradients; [likelihood-surrogate methods](#likelihood-surrogates--elbo-based-optimization) — FPO / FMPG, FPO++, Timestep Weighting, and DiPOD; [DSRL](#paper-arxiv-2506-15799) — latent steering.

<a id="science"></a>

### Scientific Discovery

Reward-guided generation of graphs, crystals, proteins, and antibodies. Predicted energy or stability provides a training signal; experimental validation remains a separate step.

<a id="core-rl-graphs-and-crystals"></a>

<a id="paper-arxiv-2608-06582"></a>

- 🚀 **[CrystalGRPO: Target-Aligned and Coverage-Preserving Reinforcement Learning for Flow-Based Crystal Structure Prediction](https://arxiv.org/abs/2608.06582)** (2026)<br>
  Jointly reinforces coordinate and lattice generation while balancing target recovery and finite-budget coverage.

<a id="paper-arxiv-2511-07158"></a>

- **[Chemeleon2 — Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning](https://arxiv.org/abs/2511.07158)** (2025) · [Code](https://github.com/hspark1212/chemeleon2)<br>
  Uses group-relative reward updates in a continuous latent crystal generator.

<a id="paper-arxiv-2511-03112"></a>

- **[MatInvent — Accelerating inverse materials design using generative diffusion models with reinforcement learning](https://arxiv.org/abs/2511.03112)** (2025) · [Code](https://github.com/schwallergroup/matinvent)<br>
  Amortizes multiobjective crystal-design rewards into a denoising policy.

<a id="paper-arxiv-2402-16302"></a>

- **[GDPO-Graph — Graph Diffusion Policy Optimization](https://arxiv.org/abs/2402.16302)** (2024) · [Code](https://github.com/sail-sg/GDPO)<br>
  Uses eager endpoint-prediction policy gradients for graph diffusion reward optimization.

<a id="broader-training-time-reward-optimization-2"></a>

<a id="preferences-reward-gradients--posterior-matching"></a>

<a id="paper-arxiv-2403-16576"></a>

- **[AbDPO — Antigen-Specific Antibody Design via Direct Energy-based Preference Optimization](https://arxiv.org/abs/2403.16576)** (2024)<br>
  Uses decomposed energy preferences for antibody sequence and structure alignment.

**See also:** [DRAKES](#paper-arxiv-2410-13643) — relaxed discrete reward gradients; [DDPP](#paper-arxiv-2410-08134) — discrete posterior matching; [Tilt Matching](#paper-arxiv-2512-21829) — particle sampling; [VIDD](#paper-arxiv-2507-00445) — value-informed distillation; [SVDD](#paper-arxiv-2408-08252) — inference-time soft-value selection.

<a id="audio-generation"></a>

### Audio

Audio-specific generation architectures, preference signals, and reward decompositions.

<a id="paper-arxiv-2511-18833"></a>

- **[PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation](https://arxiv.org/abs/2511.18833)** (2025) · [Project](https://PrismAudio-Project.github.io)<br>
  Aligns semantic, temporal, aesthetic, and spatial reasoning with corresponding audio rewards, using random-window ODE-SDE Fast-GRPO.

<a id="paper-arxiv-2504-02407"></a>

- **[F5R-TTS: Improving Flow-Matching based Text-to-Speech with Group Relative Policy Optimization](https://arxiv.org/abs/2504.02407)** (2025) · [Code](https://github.com/FrontierLabs/F5R-TTS) · [Project](https://frontierlabs.github.io/F5R/)<br>
  Reformulates flow-matching TTS outputs as Gaussian policies and applies GRPO with ASR and speaker-similarity rewards to improve intelligibility and voice cloning.

<a id="paper-arxiv-2412-21037"></a>

- **[TangoFlux / CRPO — TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and Clap-Ranked Preference Optimization](https://arxiv.org/abs/2412.21037)** (2024) · [Code](https://github.com/declare-lab/TangoFlux) · [Project](https://tangoflux.github.io/)<br>
  Uses iterative CLAP-ranked preferences to align a rectified-flow text-to-audio generator.

<a id="paper-arxiv-2404-09956"></a>

- **[Tango 2: Aligning Diffusion-based Text-to-Audio Generations through Direct Preference Optimization](https://arxiv.org/abs/2404.09956)** (2024) · [Code](https://github.com/declare-lab/tango/tree/master/tango2) · [Project](https://tango2-web.github.io/)<br>
  Fine-tunes a text-to-audio diffusion model with Diffusion-DPO on synthetic Audio-Alpaca preferences targeting sound-event presence and temporal ordering.

## Tech Reports

Model and system reports describing reward-driven training recipes for diffusion and flow models.

<a id="paper-arxiv-2606-27608"></a>

- 🚀 **[Qwen-Image-2.0-RL Technical Report](https://arxiv.org/abs/2606.27608)** (2026) · [Code](https://github.com/QwenLM/Qwen-Image) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Trains task-specific image-generation and editing policies with calibrated rewards, then merges them by on-policy velocity distillation.

<a id="paper-arxiv-2602-08676"></a>

- 🚀 **[LLaDA2.1 / EBPO — LLaDA2.1: Speeding Up Text Diffusion via Token Editing](https://arxiv.org/abs/2602.08676)** (2026) · [Code](https://github.com/inclusionAI/LLaDA2.X) · ![Subdomain: Language & Reasoning](https://img.shields.io/badge/Subdomain-Language%20%26%20Reasoning-e5e0f1.svg?style=flat&labelColor=e8ecef)<br>
  Scales block-diffusion RL through vectorized ELBO policy-ratio estimation while supporting token editing.

<a id="paper-arxiv-2511-00062"></a>

- 🚀 **[Cosmos-Predict2.5 — World Simulation with Video Foundation Models for Physical AI](https://arxiv.org/abs/2511.00062)** (2025) · [Code](https://github.com/nvidia-cosmos/cosmos-predict2.5) · ![Subdomain: World Models](https://img.shields.io/badge/Subdomain-World%20Models-ddebe8.svg?style=flat&labelColor=e8ecef)<br>
  Reward-post-trains a flow world simulator with video alignment, motion and visual-quality feedback.

<a id="paper-arxiv-2506-09113"></a>

- 🚀 **[Seedance 1.0: Exploring the Boundaries of Video Generation Models](https://arxiv.org/abs/2506.09113)** (2025) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Directly maximizes multidimensional video rewards and extends reward post-training to the accelerated diffusion refiner.

<a id="paper-arxiv-2609-08936"></a>

- **[AuK — AuK Technical Report: An Open-Source Foundational Model for Speech Generation and Editing](https://arxiv.org/abs/2609.08936)** (2026) · [Code](https://github.com/Tencent-Hunyuan/AuK) · [Project](https://auk-project.github.io/) · ![Subdomain: Audio](https://img.shields.io/badge/Subdomain-Audio-efdee6.svg?style=flat&labelColor=e8ecef)<br>
  Aligns a rectified-flow speech generator with human preference optimization for editing and reward-based RL for speech synthesis, then distills it for four-step inference.

<a id="paper-arxiv-2608-00146"></a>

- **[DiffusionGemma / SD·RL — DiffusionGemma Technical Report](https://arxiv.org/abs/2608.00146)** (2026) · ![Subdomain: Language & Reasoning](https://img.shields.io/badge/Subdomain-Language%20%26%20Reasoning-e5e0f1.svg?style=flat&labelColor=e8ecef)<br>
  Jointly improves reward and denoising efficiency through online sampler distillation with adaptive stopping.

<a id="paper-arxiv-2607-14749"></a>

- **[WanSong v1.0 Technical Report](https://arxiv.org/abs/2607.14749)** (2026) · ![Subdomain: Audio](https://img.shields.io/badge/Subdomain-Audio-efdee6.svg?style=flat&labelColor=e8ecef)<br>
  Aligns a pure diffusion song generator with DPO for global structure followed by low-noise reward gradients for detail.

<a id="paper-arxiv-2606-14409"></a>

- **[HyVLA-0.5 — Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack](https://arxiv.org/abs/2606.14409)** (2026) · [Code](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA) · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Builds a real-robot flow-VLA stack with [FlowPRO](#paper-arxiv-2606-05468) preference post-training from intervention-and-rollback data.

<a id="paper-arxiv-2602-12099"></a>

- **[GigaBrain-0.5M* / RAMP — GigaBrain-0.5M*: a VLA That Learns From World Model-Based Reinforcement Learning](https://arxiv.org/abs/2602.12099)** (2026) · [Code](https://github.com/open-gigaai/giga-brain-0) · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Conditions a flow VLA on predicted future states and value-derived improvement signals throughout iterative real-robot learning.

<a id="paper-arxiv-2602-00744"></a>

- **[ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation](https://arxiv.org/abs/2602.00744)** (2026) · [Code](https://github.com/ace-step/ACE-Step-1.5) · ![Subdomain: Audio](https://img.shields.io/badge/Subdomain-Audio-efdee6.svg?style=flat&labelColor=e8ecef)<br>
  Uses attention-alignment rewards to train the music DiT and semantic consistency rewards to train its language planner.

<a id="paper-arxiv-2601-16163"></a>

- **[Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](https://arxiv.org/abs/2601.16163)** (2026) · [Code](https://github.com/NVlabs/cosmos-policy) · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Jointly diffuses robot actions, future states and return values, then improves execution through rollout-trained model-based search.

<a id="paper-arxiv-2512-01801"></a>

- **[GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation](https://arxiv.org/abs/2512.01801)** (2025) · ![Subdomain: Robotics & Control](https://img.shields.io/badge/Subdomain-Robotics%20%26%20Control-e1ead8.svg?style=flat&labelColor=e8ecef)<br>
  Combines distributional progress filtering and latent-noise RL to specialize a flow VLA for precise long-horizon manipulation.

<a id="paper-arxiv-2509-23951"></a>

- **[HunyuanImage 3.0 Technical Report](https://arxiv.org/abs/2509.23951)** (2025) · [Code](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0) · ![Subdomain: Images & Video](https://img.shields.io/badge/Subdomain-Images%20%26%20Video-dce8f3.svg?style=flat&labelColor=e8ecef)<br>
  Combines preference, stochastic-flow and reward-distribution alignment stages in a native multimodal image generator.

<a id="paper-arxiv-2505-15809"></a>

- **[MMaDA / UniGRPO — MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809)** (2025) · [Code](https://github.com/Gen-Verse/MMaDA) · ![Subdomain: Images & Video · Language & Reasoning](https://img.shields.io/badge/Subdomain-Images%20%26%20Video%20%C2%B7%20Language%20%26%20Reasoning-e1e4f2.svg?style=flat&labelColor=e8ecef)<br>
  Unifies reward post-training of text reasoning and image-token generation in a shared discrete diffusion model.

## Further Reading & Resources

### Libraries & Toolkits

Frameworks for building and comparing reward-training systems. Paper-specific implementations are linked beside their papers.

<a id="toolkit-unirl"></a>

- **[UniRL](https://github.com/Tencent-Hunyuan/UniRL)**<br>
  A shared framework for diffusion and multimodal RL, including Flow-DPPO and other reward-training recipes.

<a id="toolkit-verl-omni"></a>

- **[VeRL-Omni](https://github.com/verl-project/verl-omni)**<br>
  Distributed multimodal post-training with separate policy, rollout, and reward roles and documented Flow-GRPO workflows.

<a id="toolkit-fastvideo"></a>

- **[FastVideo](https://github.com/hao-ai-lab/FastVideo)**<br>
  Video fine-tuning, distillation, and fast inference infrastructure for building reward-training pipelines.

<a id="toolkit-flowrl"></a>

- **[FlowRL](https://github.com/typoverflow/flow-rl)**<br>
  Modular JAX implementations and shared control benchmarks for comparing diffusion and flow policy RL.

<a id="toolkit-flow-factory"></a>

- **[Flow-Factory](https://github.com/X-GenGroup/Flow-Factory)**<br>
  Composable model, rollout, reward, advantage, and objective interfaces for image and video flow alignment.

<a id="toolkit-diffusionart"></a>

- **[DiffusionART](https://github.com/vanzll/DiffusionART)**<br>
  Black-box image and video reward optimization through aggregation-first reward tilting and drift restoration.

FlowRL here is the JAX control library (`typoverflow/flow-rl`); UniRL is Tencent-Hunyuan’s framework. Both are distinct from similarly named algorithms. FastVideo provides supporting video infrastructure.

### Background

The generative models and policy optimizer behind many methods in this list.

<a id="generative-and-rl-foundations"></a>

<a id="paper-arxiv-2303-04137"></a>

- ⭐ **[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)** (2023) · [Code](https://github.com/real-stanford/diffusion_policy) · [Project](https://diffusion-policy.cs.columbia.edu/)<br>
  Introduces conditional denoising action chunks for visuomotor behavior cloning.

<a id="paper-arxiv-2210-02747"></a>

- ⭐ **[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)** (2022)<br>
  Learns transport velocity fields by conditional regression.

<a id="paper-arxiv-2006-11239"></a>

- ⭐ **[DDPM — Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)** (2020) · [Code](https://github.com/hojonathanho/diffusion)<br>
  Establishes variational denoising training and stochastic reverse transitions.

<a id="paper-arxiv-2502-09992"></a>

- **[LLaDA — Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)** (2025) · [Code](https://github.com/ML-GSAI/LLaDA) · [Project](https://ml-gsai.github.io/LLaDA-demo/)<br>
  A masked-language diffusion backbone that underlies many later reward-training studies.

<a id="paper-arxiv-1707-06347"></a>

- **[PPO — Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)** (2017)<br>
  Clipped policy-ratio optimization, providing terminology used by later diffusion methods.

<a id="rewards--feedback-for-diffusionflow-rl"></a>

### Rewards & Feedback

The signals behind reward training: human preferences, compositional checks, video priors, latent rewards, and adaptive judges. Keep evaluation independent of the reward being optimized.

<a id="paper-arxiv-2502-01051"></a>

- ⭐ **[LPO / Latent Reward Model — Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](https://arxiv.org/abs/2502.01051)** (2025) · [Code](https://github.com/Kwai-Kolors/LPO)<br>
  Learns noise-aware rewards on diffusion latents for step-level preference optimization, avoiding repeated image decoding. `latent/process`

<a id="paper-arxiv-2305-01569"></a>

- ⭐ **[Pick-a-Pic / PickScore — Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation](https://arxiv.org/abs/2305.01569)** (2023) · [Code](https://github.com/yuvalkirstain/PickScore)<br>
  Provides open image-preference comparisons and a reusable scorer for reward-based alignment. `human preference`

- ⭐ **[ImageReward / ReFL](#paper-arxiv-2304-05977)** — Provides human-preference image scores and a reward-gradient generator update in the same paper. `human preference`

<a id="paper-arxiv-2609-36245"></a>

- 🚀 **[CoRe: Co-Evolving Reward Models for Mitigating Latent Reward Hacking in Video Diffusion Models](https://arxiv.org/abs/2609.36245)** (2026)<br>
  Co-evolves a video generator and latent reward model with a real-preference anchor to resist distributional escape and reward hacking. `adaptive/co-evolving judge` `latent/process`

<a id="paper-arxiv-2602-11146"></a>

- 🚀 **[DiNa-LRM — Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling](https://arxiv.org/abs/2602.11146)** (2026) · [Code](https://github.com/HKUST-C4G/diffusion-rm)<br>
  Uses diffusion backbones as noise-aware latent preference scorers for reward-gradient alignment and flow-policy RL. `latent/process`

<a id="paper-arxiv-2605-08703"></a>

- **[RewardHarness: Self-Evolving Agentic Post-Training](https://arxiv.org/abs/2605.08703)** (2026) · [Code](https://github.com/TIGER-AI-Lab/RewardHarness)<br>
  Adapts an agentic judge’s tools and context from preference examples, then uses its feedback to train flow-based image editing. `adaptive/co-evolving judge`
  <br><sub>The generator is trained; “training-free” refers to the judge’s weights.</sub>

<a id="paper-arxiv-2510-01544"></a>

- **[Advancing Reasoning in Diffusion Language Models with Denoising Process Rewards](https://arxiv.org/abs/2510.01544)** (2025)<br>
  Converts final-answer correctness into denoising-interval feedback for diffusion-language reasoning without training a separate process judge. `latent/process` `rule-based outcome`

- **[VideoAlign / VideoReward](#paper-arxiv-2501-13918)** — Combines multidimensional video preferences with a reward model, flow-based alignment training and a separately defined inference-guidance variant. `multidimensional video`

<a id="paper-arxiv-2312-14134"></a>

- **[Diffusion Reward: Learning Rewards via Conditional Video Diffusion](https://arxiv.org/abs/2312.14134)** (2023) · [Code](https://github.com/TEA-Lab/diffusion_reward) · [Project](https://diffusion-reward.github.io/)<br>
  Uses negative conditional entropy from an expert-video diffusion prior as a reward for a separate robot-learning agent. `robotics/video prior`

<a id="paper-arxiv-2310-11513"></a>

- **[GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment](https://arxiv.org/abs/2310.11513)** (2023) · [Code](https://github.com/djghosh13/geneval)<br>
  Checks object counts, positions, colors, and attribute binding with rules also used as compositional training rewards. `rule-based compositional`

<a id="paper-arxiv-2306-09341"></a>

- **[HPSv2 / HPDv2 — Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis](https://arxiv.org/abs/2306.09341)** (2023) · [Code](https://github.com/tgxs002/HPSv2)<br>
  Provides an image-preference dataset and scorer used as feedback for diffusion alignment. `human preference`

### Theory & Analysis

Theoretical perspectives on diffusion and flow policies, reward fine-tuning, and stochastic control.

<a id="paper-arxiv-2403-06279"></a>

- **[Fine-tuning of Diffusion Models via Stochastic Control: Entropy Regularization and Beyond](https://arxiv.org/abs/2403.06279)** (2024)<br>
  Develops a stochastic-control treatment of entropy-regularized diffusion fine-tuning and extends it to general f-divergence regularization.

<a id="paper-arxiv-2402-15194"></a>

- **[Fine-Tuning of Continuous-Time Diffusion Models as Entropy-Regularized Control](https://arxiv.org/abs/2402.15194)** (2024)<br>
  Frames continuous-time diffusion fine-tuning as entropy-regularized stochastic control against a pretrained model.

<a id="paper-arxiv-2305-13122"></a>

- **[DIPO — Policy Representation via Diffusion Probability Model for Reinforcement Learning](https://arxiv.org/abs/2305.13122)** (2023)<br>
  Represents multimodal online-RL policies with diffusion processes and introduces DIPO for model-free policy improvement.

<a id="more-paper-lists"></a>

### Related Lists

- [Awesome Diffusion Model in RL](https://github.com/opendilab/awesome-diffusion-model-in-rl) — A broader collection of diffusion policies, planning, and modeling in RL.
- [Awesome Diffusion-RL](https://github.com/YuanaHao/Awesome-Diffusion-RL) — A complementary index of diffusion and flow reward optimization.

## Citation

If this list supports your research, please cite the original papers and consider citing the repository:

```bibtex
@misc{ma2026awesomediffusionflowrl,
  author       = {Qinwei Ma},
  title        = {Awesome Diffusion \& Flow RL},
  year         = {2026},
  howpublished = {\url{https://github.com/AquaHorseM/Awesome-Diffusion-Flow-RL}},
  note         = {A curated list of reinforcement learning and reward optimization methods for diffusion and flow models}
}
```
