import torch
import numpy as np
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.rew_hidden_size = args.rew_hidden_size
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.beta = args.beta
        self.num_rew = args.num_rew
        self.action_dim = act_space.n
        self.num_agents = num_agents
        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),
                                  dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)
        self.rnn_rewards = [np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.rew_hidden_size),
            dtype=np.float32) for _ in range(self.num_rew)]

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
                                             dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.reward_pred = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self.action_dim), dtype=np.float32)
        self.act_onehot = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, self.action_dim * num_agents), dtype=np.float32)
        self.pi = np.zeros_like(self.reward_pred)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None, reward_pred=None, rnn_reward=None, pi=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        for i in range(self.num_rew):
            self.rnn_rewards[i][self.step + 1] = rnn_reward[i].copy()
        self.reward_pred[self.step] = reward_pred.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()

        self.masks[self.step + 1] = masks.copy()

        action_onehot = torch.zeros(actions.squeeze(2).shape + (self.action_dim,)).scatter_(2, torch.tensor(actions, dtype=torch.int64), 1)
        action_onehot = action_onehot.reshape(self.n_rollout_threads, 1, self.action_dim * self.num_agents).repeat(1, self.num_agents, 1).numpy()
        action_onehot = action_onehot.reshape(self.n_rollout_threads, self.num_agents, self.action_dim * self.num_agents)
        agent_mask = (1 - torch.eye(self.num_agents))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.action_dim).view(self.num_agents, -1)
        action_onehot = action_onehot * agent_mask.unsqueeze(0).numpy()
        self.act_onehot[self.step] = action_onehot.copy()
        self.pi[self.step] = pi.copy()

        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        for i in range(self.num_rew):
            self.rnn_rewards[i][0] = self.rnn_rewards[i][-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        self.value_preds[-1] = next_value
        reward_pred = self.reward_pred.copy()
        pi = self.pi.copy()
        exp_rew = (pi * reward_pred).sum(-1, keepdims = True)

        gae = 0
        baseline = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1])\
                  * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
            baseline = exp_rew[step] * self.beta  + baseline * self.gamma * self.gae_lambda * self.beta * self.masks[step + 1]
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae 
            self.returns[step] = gae - baseline  + value_normalizer.denormalize(self.value_preds[step])    

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        act_onehot = _cast(self.act_onehot)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        rewards = _cast(self.rewards)

        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 2, 0, 3, 4).reshape(-1,
                                                                                         *self.rnn_states_critic.shape[
                                                                                          3:])
        rnn_rewards = [self.rnn_rewards[i][:-1].transpose(1, 2, 0, 3, 4).reshape(-1,
                                                                        *self.rnn_rewards[0].shape[
                                                                        3:]) for i in range(self.num_rew)]
        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            rnn_reward_batchs = [[] for i in range(self.num_rew)]
            actions_batch = []
            act_onehot_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            reward_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                act_onehot_batch.append(act_onehot[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                reward_batch.append(rewards[ind:ind + data_chunk_length])
 
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
                for i in range(self.num_rew):
                    rnn_reward_batchs[i].append(rnn_rewards[i][ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)           
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            act_onehot_batch = np.stack(act_onehot_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            reward_batch = np.stack(reward_batch, axis=1)
  
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)
   
            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])
            for i in range(self.num_rew):
                rnn_reward_batchs[i] = np.stack(rnn_reward_batchs[i]).reshape(N, *self.rnn_rewards[0].shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            act_onehot_batch = _flatten(L, N, act_onehot_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            reward_batch = _flatten(L, N, reward_batch)
      
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch, reward_batch, 0, rnn_reward_batchs, act_onehot_batch

    # def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
    #     """
    #     Yield training data for MLP policies.
    #     :param advantages: (np.ndarray) advantage estimates.
    #     :param num_mini_batch: (int) number of minibatches to split the batch into.
    #     :param mini_batch_size: (int) number of samples in each minibatch.
    #     """
    #     episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
    #     batch_size = n_rollout_threads * episode_length * num_agents

    #     if mini_batch_size is None:
    #         assert batch_size >= num_mini_batch, (
    #             "PPO requires the number of processes ({}) "
    #             "* number of steps ({}) * number of agents ({}) = {} "
    #             "to be greater than or equal to the number of PPO mini batches ({})."
    #             "".format(n_rollout_threads, episode_length, num_agents,
    #                       n_rollout_threads * episode_length * num_agents,
    #                       num_mini_batch))
    #         mini_batch_size = batch_size // num_mini_batch

    #     rand = torch.randperm(batch_size).numpy()
    #     sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

    #     share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
    #     obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
    #     rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
    #     rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
    #     actions = self.actions.reshape(-1, self.actions.shape[-1])
    #     if self.available_actions is not None:
    #         available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
    #     value_preds = self.value_preds[:-1].reshape(-1, 1)
    #     returns = self.returns[:-1].reshape(-1, 1)
    #     masks = self.masks[:-1].reshape(-1, 1)
    #     active_masks = self.active_masks[:-1].reshape(-1, 1)
    #     action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
    #     advantages = advantages.reshape(-1, 1)

    #     for indices in sampler:
    #         # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
    #         share_obs_batch = share_obs[indices]
    #         obs_batch = obs[indices]
    #         rnn_states_batch = rnn_states[indices]
    #         rnn_states_critic_batch = rnn_states_critic[indices]
    #         actions_batch = actions[indices]
    #         if self.available_actions is not None:
    #             available_actions_batch = available_actions[indices]
    #         else:
    #             available_actions_batch = None
    #         value_preds_batch = value_preds[indices]
    #         return_batch = returns[indices]
    #         masks_batch = masks[indices]
    #         active_masks_batch = active_masks[indices]
    #         old_action_log_probs_batch = action_log_probs[indices]
    #         if advantages is None:
    #             adv_targ = None
    #         else:
    #             adv_targ = advantages[indices]

    #         yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
    #               value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
    #               adv_targ, available_actions_batch, 0, 0, 0, 0
