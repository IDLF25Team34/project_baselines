# Adapting Object-Centric Dexterous Manipulation Policies for Unseen Dual Manipulator Robots

This repository contains the baseline implementation and analysis for “Adapting Object-Centric Dexterous Manipulation Policies for Unseen Dual Manipulator Robots” — a project from the Intro to Deep Learning (IDL) Fall 2025 course at Carnegie Mellon University.

Our work reproduces and extends the hierarchical framework proposed in Object-Centric Dexterous Manipulation from Human Motion Data (Chen et al., NeurIPS 2024). The goal is to benchmark and adapt object-centric, human-inspired manipulation policies for novel dual-arm robotic embodiments.

## 🚀 Methodology

The project follows a hierarchical two-stage framework:

1. High-Level Wrist Planner

    - Predicts 6-DOF wrist trajectories (for left and right manipulators) from human motion capture data.

    - Trained on the ARCTIC dataset, which contains bimanual manipulation sequences with accurate hand-object motion data.

    - Implemented using the ARCTIC baseline model, which combines an encoder (CNN/MLP) with a linear prediction head to regress 3D trajectories from RGB inputs.

2. Low-Level Finger Controller

    - Takes wrist trajectories as input and generates corresponding joint trajectories for the robot fingers and wrists.

    - Trained via reinforcement learning in Isaac Gym, using the Advantage Actor-Critic (A2C) algorithm.

    - The reward function encourages accurate object tracking while minimizing control effort and contact errors.


## 🧩 Repo Structure
```bash
├── arctic # Repository for arctic dataset and high level planner baselines
│   ├── bash
│   ├── common
│   ├── docs
│   ├── LICENSE
│   ├── Makefile
│   ├── README.md
│   ├── requirements.txt
│   ├── scripts_data
│   ├── scripts_method
│   └── src
├── docker # Files for setting up docker containers for different repositories
│   ├── docker_arctic
│   └── docker_jt
├── JointTransformer # Repo for an alternate approach to high level planner
│   ├── bash
│   ├── common
│   ├── docs
│   ├── LICENSE
│   ├── Makefile
│   ├── README.md
│   ├── requirements_frozen.txt
│   ├── requirements.txt
│   ├── scripts_data
│   ├── scripts_method
│   └── src
├── ObjDexEnvs # Repo for Object-Centric Dexterous Manipulation from Human Motion Data
│   ├── assets
│   ├── dexteroushandenvs
│   ├── README.md
│   └── setup.py
└── README.md
```

## 📊 Evaluation Metrics

**Mean Per Joint Position Error (MPJPE):**
L2 distance (mm) between predicted and ground truth hand joint positions.

**Object Tracking Error:**
Euclidean distance between simulated and reference object trajectories.

**Reward Curves:**
Progression of A2C reward over training epochs for stability and convergence.

## 👥 Contributors

| Name | Affiliation | Contact |
|------|--------------|----------|
| **Ankit Aggarwal** | Robotics Institute, Carnegie Mellon University | ankitagg@andrew.cmu.edu |
| **Deepam Ameria** | Robotics Institute, Carnegie Mellon University | dameria@andrew.cmu.edu |
| **Parth Gupta** | Robotics Institute, Carnegie Mellon University | parthg@andrew.cmu.edu |
| **Shreya Shri Ragi** | Robotics Institute, Carnegie Mellon University | sragi@andrew.cmu.edu |
