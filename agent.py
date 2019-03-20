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
        self.epsilon = 0.001
        self.alpha = 0.01
        self.gamma = 1
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.policy_s = np.ones(self.nA) * self.epsilon / self.nA
        self.policy_s[np.argmax(self.Q)] = 1 - self.epsilon + (self.epsilon / self.nA)
        self.next_action = np.random.choice(np.arange(self.nA), p = self.policy_s)
        return self.next_action

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
            self.next_action = self.select_action(state)
            self.Q[state][action] += self.Q[state][action] + self.alpha * (reward + (self.gamma * self.Q[next_state][self.next_action]) - self.Q[state][action])

        if done:
            self.Q[state][action] += self.Q[state][action] + self.alpha * (reward - self.Q[state][action])
