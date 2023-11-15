import sys
from decimal import Decimal
from itertools import cycle, islice


class Node:

    def __init__(self,
                 name,
                 reward=Decimal(0),
                 node_class="chance",
                 edges=None,
                 success_rate=None,
                 val=Decimal(0),
                 edge_order=None):
        self.name = name
        self.reward = reward
        self.node_class = node_class
        self.success_rate = success_rate
        self.value = val
        self.edges = edges if edges is not None else {}
        self.edge_order = edge_order if edge_order is not None else []

    def add_edges(self, neighbors):
        self.edge_order = neighbors
        for e in neighbors:
            self.edges[e] = Decimal(0)  # default 0

    def add_probabilities(self, probabilities):
        if len(probabilities) == 1:
            success_rate = Decimal(probabilities[0])
            self.set_decision_node(success_rate)
        else:
            self.set_chance_node(probabilities)

    def set_decision_node(self, success_rate):
        self.node_class = "decision"
        self.success_rate = success_rate
        prob_for_other_edges = (1 - success_rate) / (len(self.edges) - 1)
        for i, edge in enumerate(self.edge_order):
            self.edges[edge] = success_rate if i == 0 else prob_for_other_edges

    def set_chance_node(self, probabilities):
        total_prob = sum(Decimal(p) for p in probabilities)
        if total_prob != Decimal(1):
            print("Probabilities must sum to 1")
            sys.exit(1)
        for edge, prob in zip(self.edge_order, probabilities):
            self.edges[edge] = Decimal(prob)

    def actions(self):
        for current_edge in self.edge_order:
            yield current_edge, self.get_edge_probabilities(current_edge)

    def get_edge_probabilities(self, current_edge):
        probabilities = [self.edges[edge] for edge in self.edge_order]
        return dict(zip(self.edge_order, probabilities))

    def is_decision(self):
        return self.node_class == "decision"

    def is_terminal(self):
        return self.node_class == "terminal"

    def copy(self):
        return Node(self.name, self.reward, self.node_class, self.edges, self.success_rate, self.value)
