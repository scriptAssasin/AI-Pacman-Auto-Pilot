# ergasia2_texniti

A recursive function - subfunction of getAction is created which takes as arguments a current gamestate an agent (agent is either pacman or the ghosts
if agent=0 we refer to pacman if agent >= 1 we refer to ghost) and the depth

Recursion termination condition

(The gameState.getLegalActions function returns the available actions that the agent can take)

If gameState.getLegalActions does not return any action or we reach the maximum depth then it terminates the recursion returning the value of the evaluation function (created in the first query) for the current gameState

The default case is in each call to increase the agent (not the actual one) the nextAgent variable

Because the function makes recursive calls and every time it calls itself it puts nextAgent as the second argument when we get nextAgent to be equal to the total number of agents (1 pacman + 2 ghosts) - 1 because we start from 0, then nextAgent =0 and we go back to pacman

we create the result list

finally we append