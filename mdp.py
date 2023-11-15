import random
from decimal import Decimal

from node import Node
from utils import comment, reward_line, tokenize, edge_line, probability_line


def _compute_decision_probabilities(state, action):
    return {edge: (state.success_rate if edge == action
                   else Decimal((1 - state.success_rate) / (len(state.edges) - 1)))
            for edge in state.edges}


class MDP(dict):
    class Policy(dict):
        """A class to specify the policy of an MDP. A Policy is simply a mapping between Decision nodes and Actions"""

        def __init__(self, iterable=None):
            if iterable is None:
                iterable = {}
            super(MDP.Policy, self).__init__(iterable)

        @staticmethod
        def random_policy(graph):
            """This function initializes the policy as a random policy for the given graph"""
            return MDP.Policy({node_key: random.choice(list(graph[node_key].edges.keys()))
                               for node_key in graph if graph[node_key].is_decision()})

        def __str__(self):
            policy_strings = [f"{node_key} -> {action}\n" for node_key, action in sorted(self.items())]
            return "".join(policy_strings)

        def copy(self):
            return MDP.Policy(self)

        def __eq__(self, other):
            if (not self or not other) and not (len(self) == 0 and len(other) == 0):
                return False
            for a, b in zip(self, other):
                if self[a] != other[b]:
                    return False
            return True

    def __init__(self, df=1.0, policy=None, tol=0.001, max_iter=100, use_min=False):
        super().__init__()
        self.policy = policy if policy is not None else MDP.Policy()
        self.df = df
        self.tol = tol
        self.max_iter = max_iter
        self.use_min = use_min

    def copy(self):
        new_mdp = MDP(self.df, self.policy.copy(), self.tol, self.max_iter, self.use_min)
        new_mdp.update(self)
        return new_mdp

    def mdp_solver(self):
        """Solves the MDP using value iteration and greedy policy iteration"""
        max_iterations = 100  # max_iterations depend on the input nums and
        current_policy = self.policy.copy()
        policy_changed = True  #

        for _ in range(max_iterations):
            if not policy_changed:  # policy without change, stop the loop
                break

            self._value_iteration()
            self._policy_iteration()

            previous_policy = current_policy
            current_policy = self.policy.copy()

            policy_changed = previous_policy != current_policy
            self._apply_policy(self.policy)

        self._apply_policy(self.policy)

    def _apply_policy(self, policy):
        for node in self.values():
            if node.is_decision():
                chosen_action = policy[node.name]
                other_edges = [e for e in node.edges if e != chosen_action]
                probability_for_others = (1 - node.success_rate) / len(other_edges)

                for edge in node.edges:
                    node.edges[edge] = node.success_rate if edge == chosen_action else probability_for_others

    def _policy_iteration(self):
        if not self.policy:
            self.policy = MDP.Policy.random_policy(self)

        current_policy = self.policy.copy()
        # Setting a predefined maximum number of iterations depends on the input size and the desired level of accuracy.
        max_iterations = 100

        for iteration_num in range(max_iterations):
            new_policy = current_policy.copy()

            for node in self.values():
                if node.is_decision():
                    best_action = self._find_best_action(node, new_policy)
                    if best_action:
                        new_policy[node.name] = best_action

            if new_policy == current_policy:
                break  # policy unchanged ,stop the loop

            current_policy = new_policy.copy()

        self.policy = new_policy.copy()

    def _find_best_action(self, node, policy):
        best_action = None
        best_action_value = float("-inf") if not self.use_min else float("+inf")

        for action, probabilities in node.actions():
            policy[node.name] = action
            action_value = self._value(node, policy)

            if (best_action_value < action_value and not self.use_min) or (
                    best_action_value > action_value and self.use_min):
                best_action_value = action_value
                best_action = action

        return best_action

    def _value_iteration(self):

        current_values = {node_name: node_value.value for node_name, node_value in self.items()}
        if not self.policy:
            self.policy = MDP.Policy.random_policy(self)

        for iteration_num in range(int(self.max_iter)):
            new_values = {node.name: self._value(node, self.policy) for node in self.values()}

            # Checking whether values have converged.
            values_converge = all(abs(current_values[node_name] - new_value) <= self.tol
                                  for node_name, new_value in new_values.items())

            if values_converge:
                break

            # update current value
            for node_name, new_value in new_values.items():
                self[node_name].value = new_value

            current_values = new_values.copy()

    def _value(self, state, policy=None):
        policy = policy or self.policy
        action = policy[state.name] if state.is_decision() else None

        # decision node to calculate choose probabilities
        probabilities = (_compute_decision_probabilities(state, action)
                         if state.is_decision()
                         else state.edges)

        # calculate expected_utility
        expected_utility = sum(Decimal(self.df) * Decimal(prob) * self[node_name].value
                               for node_name, prob in probabilities.items())

        return Decimal(state.reward) + expected_utility

    def mdp_result(self):
        print(str(self.policy))

        # format list for node value
        values = ["{}={:.3f}".format(node_name, round(node.value, 3)) for node_name, node in sorted(self.items())]

        # join value and print
        print(" ".join(values))

    @staticmethod
    def read_inputfile(file_name, df=1.0, tol=0.01, max_iter=100, use_min=False):
        mdp_instance = MDP(df, None, tol, max_iter, use_min)
        with open(file_name) as in_file:
            lines = in_file.readlines()
        MDP._parse_input(mdp_instance, lines=lines)
        mdp_instance.policy = MDP.Policy.random_policy(mdp_instance)
        mdp_instance._apply_policy(mdp_instance.policy)
        return mdp_instance

    @staticmethod
    def _parse_input(output_mdp, lines):
        unclaimed_probability_lines = []

        for line in lines:
            line = line.strip()
            if not line or comment.match(line):
                continue

            node_name, data = tokenize(line)

            if reward_line.match(line):
                output_mdp.setdefault(node_name, Node(node_name)).reward = Decimal(data)

            elif edge_line.match(line):
                output_mdp.setdefault(node_name, Node(node_name)).add_edges(data)

            elif probability_line.match(line):
                if node_name in output_mdp:
                    output_mdp[node_name].add_probabilities(data)
                else:
                    unclaimed_probability_lines.append(line)

        while unclaimed_probability_lines:
            line = unclaimed_probability_lines.pop(0)
            node_name, probabilities = tokenize(line)
            if node_name in output_mdp:
                output_mdp[node_name].add_probabilities(probabilities)

        for node_name, node in output_mdp.items():
            if not node.edges:
                node.node_class = "terminal"
            elif all(prob == 0 for prob in node.edges.values()) and not node.is_terminal():
                if len(node.edges) != 1:
                    node.node_class = "decision"
                    node.success_rate = 1
                for i, edge in enumerate(node.edges):
                    node.edges[edge] = 1 if i == 0 else 0
