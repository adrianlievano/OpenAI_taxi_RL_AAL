import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        #self.epsilon = 0.001
        self.alpha = 0.01
        self.gamma = .5
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def epsilon_greedy_probs(self, Q_s, i_episode, eps = None):
            epsilon = 1.0 / i_episode
            if eps is not None:
                epsilon = eps
            policy_s = np.ones(self.nA) * epsilon / self.nA
            policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
            return policy_s

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.i_episode = i_episode
        self.policy_s = self.epsilon_greedy_probs(self.Q, i_episode)
        action = np.random.choice(np.arange(self.nA), p = self.policy_s)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            policy_s = self.epsilon_greedy_probs(self.Q[next_state], self.i_episode)
            next_action = self.select_action(next_state, self.i_episode)
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + (self.gamma * self.Q[next_state][next_action]) - self.Q[state][action])

        if done:
            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward - self.Q[state][action])
