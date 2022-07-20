from .environment import MultiAgentEnv
from .scenarios import load
import numpy as np
from gym import spaces

def MPEEnv(args):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world,
                        scenario.reward, scenario.observation, scenario.info)

    return env


class Matrix:
    def __init__(self, args):
        n = 16
        self.num_agent = n
        self.max_step = n
        self.action_dim = n
        self.observation_space = [[self.max_step] for _ in range(self.num_agent)]
        self.share_observation_space = [[self.max_step * self.num_agent] for _ in range(self.num_agent)]
        self.steps = 0
        self.action_space = [spaces.Discrete(self.action_dim) for _ in range(self.num_agent)]
       
    def seed(self,seed):
        return 0
        
    def observation(self):
        obs = np.zeros(self.max_step)
        obs[self.steps] = 1
        return np.array([obs for _ in range(self.num_agent)])

    def step(self, action):
        self.steps += 1
        rew_n = self.reward(action)
        obs_n = self.observation()
        done_n = self.done()
        info_n = None
        return obs_n, rew_n, done_n, info_n

    def reset(self):
        self.steps = 0
        return self.observation()

    def reward(self, action):
        action = np.argmax(action, axis=1)
        rew = (action.reshape(-1)[self.steps % self.num_agent] - (self.action_dim-1)/2)* np.power(-1, self.steps)
        return  np.array([[rew] for _ in range(self.num_agent)])

    def done(self):
        if self.steps == self.max_step - 1:
            return [True for _ in range(self.num_agent)]
        else:
            return [False for _ in range(self.num_agent)]
