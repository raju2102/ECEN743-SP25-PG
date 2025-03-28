# ECEN743-SP25-PG

## Overview

1. You have to submit a PDF report, your code, and a video to Canvas.
2. Put all your files (PDF report, code, and video) into a **single compressed folder** named `Lastname_Firstname_A5.zip`.
3. Your PDF report should include answers and plots to all the questions.

## General Instructions

1.  This assignment must be completed on the TAMU HPRC. Apply for an account [here](https://hprc.tamu.edu/).
1.  You will complete this assignment in a Python (.py) file. Please check `PG.py` for the starter code.
1.  Type your code between the following lines
    ```
    ###### TYPE YOUR CODE HERE ######
    #################################
    ```
1. You do not need to modify the rest of the code for this assignment. However, feel free to do so if needed. The default hyperparameters should be able to solve LunarLander-v3.
1. The problem is solved if the **total reward per episode** is 200 or above. *Do not* stop training on the first instance your agent gets a reward above 200, your agent must achieve a reward of 200 or above consistently.
1. The x-axis of your training plots should be  training episodes (or training iterations), and the y-axis should be episodic reward (or average episodic reward per iteration). You may have to use a sliding average window to get clean plots.
1. **Video Generation:** You do not have to write your own method for video generation. Gymnasium has a nice, self-containted wrapper for this purpose. Please read more about Gymnasium wrappers [here](https://gymnasium.farama.org/api/wrappers/).

## Problems

In this homework, you will train a policy gradient algorithm to land a lunar lander **with continuous  actions** on the surface of the moon. We will use the Lunar Lander environment (LunarLander-v3) from  Gymnasium. The environment consists of a lander with continuous  actions and a continuous state space. A detailed description of the environment can be found [here](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

1. **REINFORCE Algorithm:** Implement Eq.1. You should include the training curve in the PDF (x-axis is the number of episode and the y-axis should be episodic undiscounted-cumulative reward). Use a sliding window average to get smooth plots. Include a description of the hyperparameters used. Try to find the optimal hyperparameters that will enable fast convergence to the optimal policy.  

2. **Policy Gradient Algorithm:** Implement Eq.2 and produce similar results as above.

3. **Policy Gradient with Baseline:** Implement Eq.3 and produce similar results as above. Also, you should submit a video of the smooth landing achieved by your RL algorithm. Video is needed only for this part.

4. **Another Environmnet:** Now, learn the optimal policy for another control problem (environment) using policy gradient algorithm. You can select one environment from the Classical Control set, Box2D set or Atari Games set in Gymnasium. In the PDF, you need to clearly specify the environment and provide a link to the corresponding page in Gymnasium. You need to include the training curve and describe the hyperparameters used. You should also include the video of the performance.

## HPRC Intructions

### Installation
```
cd $SCRATCH
ml GCCcore/13.3.0
ml git/2.45.1
ml Miniconda3/23.10.0-1
conda init
source ~/.bashrc
source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
git clone https://github.com/ECEN743-TAMU/ECEN743-SP25-PG 
cd ECEN743-SP25-PG 
conda create -n rl_env python=3.11
conda activate rl_env
pip install swig torch
pip install gymnasium[other]
pip install gymnasium[box2d]
```

### Running
After making required changes to `PG.py` and `run.slurm` 
```
conda deactivate
sbatch run.slurm
```
