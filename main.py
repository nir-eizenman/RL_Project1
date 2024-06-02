import gym
import torch

print(torch.__version__)
print(gym.__version__)

# Question 1 (Longest Common Subsequence)
"""
Write a dynamic programming solution for the following problem.

Given: Two sequences (or strings) X(1 : m) and Y (1 : n).

Goal: Return the length of the longest common subsequence of both X and Y (not
necessarily contiguous).

For Example:

    X = AVBVAMCD
    Y = AZBQACLD

Answer = 5

* For full credits, your algorithm should run in time O(nm).
"""


# Answer 1
def longest_common_subsequence(X, Y):
    # this is a 2D dynamic programming problem, we'll create a 2D array to store the results (memoization)
    m = len(X)
    n = len(Y)
    # initialize a 2D array of size (m+1) x (n+1) filled with zeros
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # fill the array with the results of the subproblems from end to start
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            # if the characters match, we add 1 to the result of the subproblem
            if X[i] == Y[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            # otherwise, we take the maximum of the results of the subproblems (either by removing a character from X or Y)
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


# time complexity: O(nm) because do a constant amount of work for each cell in the 2D array of size (m+1)*(n+1)

X = "AVBVAMCD"
Y = "AZBQACLD"
print(longest_common_subsequence(X, Y))  # expected: 5

# Question 2 (Moses the mouse)
"""
Moses the mouse starts his journey at the south west room in a M*N rectangular apartment
with M*N rooms of size 1*1, some of which contain cheese. After his rare head injury in
the mid-scroll war, Moses can only travel north or east. An illustration of Moses’s life for
M = 5, N = 8 is given in the following figure:

Being a mouse and all, Moses wants to gather as much cheese as possible until he reaches
the north-east room of the apartment.

1. Formulate the problem as a finite horizon decision problem: Define the state space,
    the action space and the cumulative cost function.
    
2. What is the horizon of the problem?

3. How many possible trajectories are there? How does the number of trajectories behaves
    as a function of N when M = 2? How does it behave as a function of N when M = N?
    please note that you don’t need to calculate the exact number of states, you can give
    the order number (this also apply to the rest of this question).
    
4. Aharon, Moses’s long lost war-buddy woke up confused next to Moses and decided to
    join him in his quest (needless to say, both mice suffer the same rare head injury).
    
    (a) Explain what will happen if both mice ignore each other’s existence and act
    ’optimal’ with respect to the original problem.
    
    (b) Assume both mice decided to coordinate their efforts and split the loot. How
    many states and actions are there now?
    
    (c) Now their entire rarely-head-injured division has joined the journey. Assume
    there’s a total of K mice, how many states and actions are there now?
"""

# Answer 2
"""
1. The state space is the set of all possible positions in the apartment. The action space is the set of all possible
    movements (north or east). The cumulative cost function is the sum of the costs of the movements.
    S = { (i, j) ∶ 1≤i≤M, 1≤j≤N }
    A = { north, east } (note that when by the wall, only one action is available: { north } or { east } or { none } if by the corner)
    R(mouse_path) = ∑ t = 1 to T C(s_t, a_t) + R(s_T) where C(s_t, a_t) is the cost of moving from state s_t to state s_t+1
    and R(s_T) is the reward of reaching state s_T (1 if there's cheese, 0 otherwise).
    
2. The horizon of the problem is the number of steps needed to reach the north-east room from the south-west room,
    which is M + N - 2 (M-1 steps to the north and N-1 steps to the east).
    
3. The number of possible trajectories is the number of ways to choose M-1 steps to the north out of M+N-2 steps
    (or equivalently, the number of ways to choose N-1 steps to the east out of M+N-2 steps), this is the number of 
    trajectories because we can only move north or east at each step at a set amount of steps and we just need to
    choose the order of the steps.
    This is given by: (M+N-2 choose M-1) or (M+N-2 choose N-1).
    
    When M = 2, the number of trajectories is (2+N-2 choose N-1) = (N choose 1) = N.
    When M = N, the number of trajectories is (2N-2 choose N-1).

4.a. If both mice ignore each other's existence and act 'optimal' with respect to the original problem, they will
     each take the path that maximizes their own cheese collection without considering the other mouse's path,
     this might lead to suboptimal results for the group, and both mice will collect the same amount of cheese together
     as if they were alone (they will share the same path and the same amount of cheese between them).
     
4.b. In this case we get: S = { (i_1, j_1, i_2, j_2) ∶ 1≤i_1≤M, 1≤j_1≤N, 1≤i_2≤M, 1≤j_2≤N }, the size of the state space
     is M*N*M*N = M^2 * N^2.
     The action space is A = { (north, north), (north, east), (east, north), (east, east) }.
     As with the original problem, some actions might not be available if the mice are by the wall or corner, in
     which case one of both of their movements might be limited, which reduces the number of actions available for one
     or both mice.
     
4.c. In this case we get: S = { (i_1, j_1, i_2, j_2, ..., i_K, j_K) ∶ 1≤i_1≤M, 1≤j_1≤N, ..., 1≤i_K≤M, 1≤j_K≤N },
     the size of the state space will be M^N * N^K.
     The action space will be A={ (north, north, ..., north), (north, north, ..., east), ..., (east, east, ..., east) }.
     The number of actions is 2^K (similar to the binary representation of the actions, where 0 means north and
     1 means east). And again, some actions might not be available if the mice are by the wall or corner.
"""
