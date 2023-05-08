# RUBICON

## Introduction
Rule-based control (RBC) is widely adopted in buildings due to its stability and
    robustness. It resembles a behavior cloning methodology refined by human experts; however,
    it is incapable of adapting to distribution drifts.
        Reinforcement learning (RL) can adapt to changes but needs to 
    learn from scratch in the online setting. On the other hand, the learning ability is limited in offline settings
    due to extrapolation errors caused by selecting out-of-distribution actions.
        In this paper, we explore how to incorporate RL with a rule-based control policy to combine 
    their strengths to continuously learn a scalable and robust policy in both 
    online and offline settings. 
        We start with representative online and offline RL methods, TD3 and TD3+BC,
    respectively. Then, we develop a dynamically weighted actor loss function to 
    selectively choose which policy for RL models to learn from at each training iteration. 
        With extensive experiments across various weather conditions in both deterministic and 
    stochastic scenarios, we demonstrate that our algorithm, 
    <ins>ru</ins>le-<ins>b</ins>ased <ins>i</ins>ncorporated
    <ins>con</ins>trol regularization (RUBICON), outperforms state-of-the-art
    methods in offline settings by $40.7\%$ and improves the baseline method by $49.7\%$ in online settings with respect to a reward consisting of thermal comfort and energy consumption in building-RL environments. 

## How to run it
1. Successfully install [Sinergym](https://github.com/ugr-sail/sinergym)
2. Git clone our repository ```git clone https://github.com/HYDesmondLiu/RUBICON.git```
3. ```cd ./RUBICON/01_BRL/``` or ```cd ./RUBICON/02_OnlineRL/```
4. Modify the ```Sinergym*.py``` to fit your GPU availability.
5. Run ```python Sinergym_BRL.py``` or ```python Sinergym.py```

## Building BRL Dataset
- The dataset we learn from for the offline approach is at https://github.com/HYDesmondLiu/B2RL


## Cite our paper

--- Will update after the full paper is published officially---
