# Implementation of the algorithms described by Dr. Dana Angluin in her paper Inference of Reversible Languages (1982)

# Infers (finite state) canonical acceptor for the smallest k-reversible language that includes the input S.
# On input S, k-RI constructs the prefix-tree acceptor for S, A0 = PT(S),
# and finds the finest partition pi_f of the states of A0 such that A0/pi_f is k-reversible.
# It then outputs a canonical acceptor for L(A0/pi_f) and halts.
#
# Input:
# S - a list of strings in the language to be inferred
# k - int denoting k-reversibility
import copy
import numpy as np
import fst


def k_RI(S, k):
    # if S is empty, output the empty acceptor an halt
    if len(S) == 0:
        print("Sample is empty")
        return PT_Acceptor(S)

    # if k=0 use zero reversible
    if k == 0:
        return zero_RI(S)

    ############# Initialization ##################

    A0 = PT_Acceptor(S)  # A0 is the prefix tree acceptor for S
    partition = Partition(
        [[n] for n in A0.nodes])  # holds all the different partitions of A0 states - first the trivial partition
    quotient_acceptor = Quotient_Acceptor(A0, copy.deepcopy(partition))

    i = 0

    ########### Merging Loop ####################
    while True:

        ####### Merging Data Initialization #########
        quotient_acceptor = Quotient_Acceptor(A0, copy.deepcopy(partition))

        H = [((x_0, x_1), (y_0, y_1)) for (t0, x_0, y_0) in quotient_acceptor.edges for (t1, x_1, y_1) in
             quotient_acceptor.edges if t0 == t1]

        C = [[[1 for y in quotient_acceptor.nodes] for x in quotient_acceptor.nodes]]  # list of matrices - first all 1s

        for j in range(1, k + 2):
            c = [[0 for y in quotient_acceptor.nodes] for x in quotient_acceptor.nodes]

            c_prev = C[j - 1]
            for (x_0, x_1), (y_0, y_1) in H:
                if c_prev[x_0][x_1] == 1:
                    c[y_0][y_1] = 1
            C.append(c)

        ########## Merging ###################
        next_partition = Partition(copy.deepcopy(partition.blocks))  # the next parition
        c = C[k]
        merged = False  # used to indicate if a merger has occurred

        if not merged:
            for (t0, x_0, y_0) in quotient_acceptor.edges:
                if not merged:
                    for (t1, x_1, y_1) in quotient_acceptor.edges:
                        if t0 == t1 and x_0 == x_1 and y_0 != y_1:
                            next_partition.merge_blocks(y_0, y_1)
                            merged = True
                            break

                        if t0 == t1 and y_0 == y_1 and x_0 != x_1:
                            if c[x_0][x_1] == 1:
                                next_partition.merge_blocks(x_0, x_1)
                                merged = True
                                break

        if not merged:
            for x in quotient_acceptor.nodes:
                if not merged:
                    for y in quotient_acceptor.nodes:
                        if x != y:
                            if c[x][y] == 1:
                                next_partition.merge_blocks(x, y)
                                merged = True
                                break

        # Check if no further merges occurred else add next parition to partitions list, update i and continue
        if not merged:
            break
        else:
            i += 1
            partition = next_partition

    ############# Termination #######################

    A = Quotient_Acceptor(A0, partition)
    return minimize(A, get_alphabet(S))


# The Zero Reversible Inference algorithm takes as input a finite non-empty set of strings S
# and outputs a particular deterministic acceptor A = ZR(S)
# such that L(A) is the smallest zero-reversible language that contains S
def zero_RI(S):
    ############# Initialization ##################
    A0 = PT_Acceptor(S)  # A0 is the prefix tree acceptor for S
    partition = Partition(
        [[n] for n in A0.nodes])  # holds all the different partitions of A0 states - first the trivial partition
    alphabet = get_alphabet(S)
    reversed_edges = A0.get_reverse_edges()

    s = {(tuple([q]), b): A0.edges[(b, q)] if (b, q) in A0.edges else None for b in alphabet for q in
         A0.nodes}  # dictionary that holds b-successors
    p = {(tuple([q]), b): reversed_edges[(b, q)] if (b, q) in reversed_edges else None for b in alphabet for q in
         A0.nodes}  # dictionary that holds b-predecessors

    q0 = A0.is_accepting[0]

    to_merge = [(q0, A0.is_accepting[q]) for q in range(1, len(A0.is_accepting))]

    i = 0

    while len(to_merge) != 0:
        (q1, q2) = to_merge.pop(0)

        b1 = partition.get_block_index_of_element(q1)
        b2 = partition.get_block_index_of_element(q2)

        block1 = partition.blocks[b1]
        block2 = partition.blocks[b2]

        if block1 != block2:
            next_partition = Partition(copy.deepcopy(partition.blocks))  # the next partition
            next_partition.merge_blocks(b1, b2)
            partition = next_partition

            for b in alphabet:
                update(block1, block2, b, s, to_merge)  # s-update
                update(block1, block2, b, p, to_merge)  # p-update

            i += 1

    ############### Termination ##########
    return Quotient_Acceptor(A0, partition)


# Note: s_p is the dictionary (b-successors or b-predecessors)
# It places (s_p(B1, b), s_p(B2, b)) on to_merge if:
# both s_p(B1, b) and s_p(B2, b) are nonempty
# Otherwise:
# it defines s_p(B3 = B1 union B2, b) = s_p(B1,b) if this is nonempty or s_p(B2, b) if that is nonempty
def update(block1, block2, b, s_p, to_merge):
    block3 = list(set(block1 + block2))
    if (tuple(block1), b) in s_p and s_p[(tuple(block1), b)] is not None:
        if (tuple(block2), b) in s_p and s_p[(tuple(block2), b)] is not None:
            to_merge.append((s_p[(tuple(block1), b)], s_p[(tuple(block2), b)]))
            s_p[(tuple(block3), b)] = s_p[(tuple(block1), b)]

        else:
            s_p[(tuple(block3), b)] = s_p[(tuple(block1), b)]

    elif (tuple(block2), b) in s_p and s_p[(tuple(block2), b)] is not None:
        s_p[(tuple(block3), b)] = s_p[(tuple(block2), b)]

    else:
        s_p[(tuple(block3), b)] = None


# Returns the alphabet (i.e. the list of words) from a list of strings S
def get_alphabet(S):
    return sorted(list(set([t for s in S if s != "" or s != '' for t in s.split(" ")])))


# DFA Minimization using Equivalence Theorem
def minimize(A, alphabet):
    # format A's edges into dictionary form
    edges = {}
    for (b, x, y) in A.edges:
        edges[(b, x)] = y

    partitions = []

    # create parition0 = {Accepting, Q - Accepting}
    blocks = [[], []]
    for n in A.nodes:
        # if n is accepting then place it in blocks[0]
        if n in A.is_accepting:
            blocks[0].append(n)
        # if n is not accepting then place it in blocks[1]
        else:
            blocks[1].append(n)
    partitions.append(Partition(blocks))

    k = 0

    while True:
        partition = partitions[k]
        blocks = partition.blocks
        next_blocks = []

        # for each block in partitions[k] divide it into subsets such that two states x, y
        # are in the same subset iff for each input symbol p and 1 make trantion to the states
        # of the same set of partitions[k]
        for block in blocks:
            equiv = {}  # if (x, y) = True then x, y are equivalent, if (x, y) = false then they are not
            for x in block:
                for y in block:
                    if x != y:
                        if (y, x) in equiv:
                            continue
                        equiv[(x, y)] = True
                        for b in alphabet:
                            if (b, x) in edges and (b, y) in edges:
                                x_prime = edges[(b, x)]
                                y_prime = edges[(b, y)]
                                block_x = partition.get_block_index_of_element(x_prime)
                                block_y = partition.get_block_index_of_element(y_prime)
                                if block_x != block_y:
                                    equiv[(x, y)] = False
                            elif (b, x) in edges and (b, y) not in edges:
                                equiv[(x, y)] = False
                            elif (b, x) not in edges and (b, y) in edges:
                                equiv[(x, y)] = False

            subsets = []
            if len(block) == 1:
                subsets.append(block)
            else:
                for (x, y) in equiv:
                    if equiv[(x, y)]:
                        added = False
                        s = 0
                        while True:
                            if s >= len(subsets):
                                break
                            if s > len(subsets) - 1:
                                break
                            sub = subsets[s]
                            if x in sub:
                                combined = False
                                for s_prime in range(len(subsets)):
                                    sub_prime = subsets[s_prime]
                                    if s_prime != s and y in sub_prime:
                                        combined_set = sub + sub_prime
                                        combined = True
                                        if s > s_prime:
                                            subsets.pop(s)
                                            subsets.pop(s_prime)
                                        else:
                                            subsets.pop(s_prime)
                                            subsets.pop(s)
                                        subsets.append(combined_set)
                                        s -= 1
                                        break
                                if not combined:
                                    sub.append(y)
                                    added = True
                            elif y in sub:
                                combined = False
                                for s_prime in range(len(subsets)):
                                    sub_prime = subsets[s_prime]
                                    if s_prime != s and x in sub_prime:
                                        combined_set = sub + sub_prime
                                        combined = True
                                        if s > s_prime:
                                            subsets.pop(s)
                                            subsets.pop(s_prime)
                                        else:
                                            subsets.pop(s_prime)
                                            subsets.pop(s)
                                        subsets.append(combined_set)
                                        s -= 1
                                        break
                                if not combined:
                                    sub.append(x)
                                    added = True
                            s += 1
                        if not added:
                            subsets.append([x, y])

            for x in block:
                exists = False
                for sub in subsets:
                    if x in sub:
                        exists = True
                if not exists:
                    subsets.append([x])

            for sub in subsets:
                next_blocks.append(list(set(sub)))

        partitions.append(Partition(next_blocks))
        k += 1

        if np.array_equal(next_blocks, blocks):
            break

    f = k

    # Create acceptor equivalent to A but with edges in format (b, x) = y because k_RI outputs it on the form (b, x, y)
    A = Acceptor(A.nodes, edges, A.is_accepting, A.initial)

    minimized = Quotient_Acceptor(A, partitions[f])

    return minimized


# Simple Acceptor Class
class Acceptor:

    def __init__(self, nodes, edges, is_accepting, initial):
        self.nodes = nodes
        self.edges = edges
        self.is_accepting = is_accepting
        self.initial = initial
        pass


# Prefix Tree Acceptor Class
# It computes the prefix tree from a list of strings S
# nodes - list of nodes numerated with integers
# edges - dictionary of the form (word, state) = state'
# is accepting - list with accepting nodes
class PT_Acceptor:

    def __init__(self, S):
        self.nodes = [0]
        self.edges = {}
        self.is_accepting = []
        self.initial = 0
        self.get_prefix_tree(S)

    def get_prefix_tree(self, S):
        S = sorted(S)

        # if empty string part of the language mark final state as empty
        # if '' in S:
        #   self.is_accepting.append(0)

        for s in S:
            if s is '':
                continue
            tokens = s.split(" ")
            current_node = 0
            for i in range(len(tokens)):
                t = tokens[i]

                # if there exists an edge already update current_node
                if (t, current_node) in self.edges:
                    current_node = self.edges[(t, current_node)]

                else:
                    new_node = len(self.nodes)  # create a new node
                    self.nodes.append(new_node)  # append the node
                    self.edges[(t, current_node)] = new_node  # update the edges dictionary
                    current_node = new_node  # update the current node

                # if t is the last word in the sentence s mark the node reached as final if not already
                if i == len(tokens) - 1 and current_node not in self.is_accepting:
                    self.is_accepting.append(current_node)

    # used in zero-reversible inference
    def get_reverse_edges(self):
        return {(t, self.edges[(t, n)]): n for (t, n) in self.edges}


# Quotient Acceptor Class
# It computes the quotient A0/pi of a given (prefix tree) acceptor A0 and a partition pi of A0's nodes
# Let A(Q, I , F, delta) be any acceptor. If pi is any partition of Q,
# we define the quotient acceptor A/pi = (Q', I', F', delta') as follows:
# Q' is the set of blocks of pi.
# I' is the set of blocks of pi that contain any state in I
# F' is the set of all blocks of pi that contain an element of F
# The block B2 is in delta'(B1, a) whenever there exist q1 in B1 and q2 in B2 s.t. q2 in delta(q1, a)
class Quotient_Acceptor:

    def __init__(self, A0, pi):
        self.A0 = A0
        self.pi = pi
        self.nodes = self.get_nodes()
        self.edges = self.get_edges()
        self.is_accepting = self.get_accepting()
        self.initial = self.get_initial()

    def get_nodes(self):
        return [n for n in range(len(self.pi.blocks))]

    def get_edges(self):
        return list(set(
            [(t, self.pi.get_block_index_of_element(node), self.pi.get_block_index_of_element(self.A0.edges[(t, node)]))
             for (t, node) in self.A0.edges]))

    def get_accepting(self):
        return list(set([self.pi.get_block_index_of_element(a) for a in self.A0.is_accepting]))

    def get_initial(self):
        init = self.A0.initial
        blocks = self.pi.blocks
        for i in range(len(blocks)):
            if init in blocks[i]:
                return i

    def print_me(self):
        print("NODES")
        print(self.nodes)
        print(" ")

        print("INITIAL")
        print(self.initial)
        print(" ")

        print("ACCEPTING")
        print(self.is_accepting)
        print(" ")

        print("EDGES")
        print(self.edges)
        print(" ")

    # Return equivalent OpenFsm Object
    def to_fsm(self):
        f = fst.Fst()
        states = {}
        # create all states
        for n in self.nodes:
            states[n] = f.add_state()
        # set initial state
        f.set_start = states[self.initial]
        # set final states
        for n in self.is_accepting:
            f.set_final(states[n])
        # create all edges
        for (t, u, v) in self.edges:
            f.add_arc(u, v, t, 1)

        return f


# Class to store and manipulate partitions
class Partition:
    def __init__(self, blocks):
        self.blocks = blocks

    def get_block_index_of_element(self, element):
        for i in range(len(self.blocks)):
            if element in self.blocks[i]:
                return i
        return None

    def get_block_index(self, block):
        for i in range(len(self.blocks)):
            if block == self.blocks[i]:
                return i

    def merge_blocks(self, index1, index2):
        if index1 < index2:
            block1 = self.blocks.pop(index1)
            block2 = self.blocks.pop(index2 - 1)
        else:
            block2 = self.blocks.pop(index2)
            block1 = self.blocks.pop(index1 - 1)

        block_merged = list(set(block1 + block2))
        self.blocks.append(block_merged)
        return