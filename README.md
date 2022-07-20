# DAE

This codebase accompanies paper "Difference Advantage Estimation for Multi-Agent Policy Gradients"([link](https://proceedings.mlr.press/v162/li22w/li22w.pdf)). The implementation is based on [MAPPO](https://github.com/marlbenchmark/on-policy) codebase. 

## Environments Supported

- [StarCraftII(SMAC)](https://github.com/oxwhirl/smac)
- [Multiagent Particle-World Environment (MPE)](https://github.com/openai/multiagent-particle-envs)
- [Matrix Game](https://proceedings.mlr.press/v162/li22w/li22w.pdf)

## Installation instructions

Please follow the instructions in [MAPPO](https://github.com/marlbenchmark/on-policy) codebase.

## Run an experiment 

Here we use train_mpe.sh as an example:
```shell
cd onpolicy/scripts
chmod +x ./train_mpe.sh
./train_mpe.sh
```
