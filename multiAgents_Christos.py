# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foods = newFood.asList()
        evaluation = 0
        for i in range(len(newGhostStates)):
            ghostPosition = childGameState.getGhostPosition(i+1)
            distance = util.manhattanDistance(newPos,ghostPosition)
            if distance < 2:
                evaluation -= 2
            if childGameState.hasWall(newPos[0],newPos[1]):
                evaluation -= 2
            if newPos in foods:
                evaluation += 1
            foodDistance = []
            for food in foods:
                foodDistance.append(util.manhattanDistance(food,newPos))
            if foodDistance:
                evaluation -= 0.5*min(foodDistance)
        return childGameState.getScore() + evaluation



def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        return self.Minimax(gameState)

    def TerminalTest(self,gameState):
        if gameState.isWin() or gameState.isLose():
            return True
        return False

    def Minimax(self,gameState):
        decision = Directions.STOP
        arg_max = -float('inf')
        for Action in gameState.getLegalActions(0):
            temp_max = self.MinValue(gameState.getNextState(0,Action),0,1)
            if temp_max > arg_max:
                arg_max = temp_max
                decision = Action
        return decision

    def MinValue(self, gameState, currentDepth, ghostIndex):
        if self.TerminalTest(gameState):
           return self.evaluationFunction(gameState)
        value = float('inf')
        for Action in gameState.getLegalActions(ghostIndex):
            if ghostIndex == gameState.getNumAgents() - 1:
                value = min(value,self.MaxValue(gameState.getNextState(ghostIndex,Action),currentDepth+1)) #?
            else:
                value = min(value,self.MinValue(gameState.getNextState(ghostIndex,Action),currentDepth,ghostIndex+1))
        return value

    def MaxValue(self, gameState, currentDepth):
        if self.TerminalTest(gameState) or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        value = -float('inf')
        for Action in gameState.getLegalActions(0):
            value = max(value,self.MinValue(gameState.getNextState(0,Action),currentDepth,1))
        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        return self.Minimax_A_B(gameState)

    def TerminalTest(self,gameState):
        if gameState.isWin() or gameState.isLose():
            return True
        return False

    def Minimax_A_B(self,gameState):
        return self.MaxValue(gameState,0,-float('inf'),float('inf'))

    def MinValue(self, gameState, currentDepth, ghostIndex, a, b):
        if self.TerminalTest(gameState):
           return self.evaluationFunction(gameState)
        value = float('inf')
        for Action in gameState.getLegalActions(ghostIndex):
            if ghostIndex == gameState.getNumAgents() - 1:
                value = min(value,self.MaxValue(gameState.getNextState(ghostIndex,Action),currentDepth+1,a,b)) #?
            else:
                value = min(value,self.MinValue(gameState.getNextState(ghostIndex,Action),currentDepth,ghostIndex+1,a,b))
            if value < a:
                return value
            b = min(b,value)
        return value

    def MaxValue(self, gameState, currentDepth, a, b):
        move = Directions.STOP
        if self.TerminalTest(gameState) or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        value = -float('inf')
        for Action in gameState.getLegalActions(0):
            temp = self.MinValue(gameState.getNextState(0,Action),currentDepth,1,a,b)
            if temp > value:
                value = temp
                move = Action
            if value > b:
                return value
                move = Action
            a = max(a,value)
        if currentDepth == 0:
            return move
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.Expectimax(gameState)
        #util.raiseNotDefined()

    def TerminalTest(self,gameState):
        if gameState.isWin() or gameState.isLose():
            return True
        return False

    def Expectimax(self,gameState):
        decision = Directions.STOP
        arg_max = -float('inf')
        for Action in gameState.getLegalActions(0):
            temp_max = self.ExpectedValue(gameState.getNextState(0,Action),0,1)
            if temp_max > arg_max:
                arg_max = temp_max
                decision = Action
        return decision

    def ExpectedValue(self, gameState, currentDepth, ghostIndex):
        if self.TerminalTest(gameState):
            return self.evaluationFunction(gameState)
        value = 0
        for Action in gameState.getLegalActions(ghostIndex):
            if ghostIndex == gameState.getNumAgents() - 1:
                value += self.MaxValue(gameState.getNextState(ghostIndex,Action),currentDepth+1) / len(gameState.getLegalActions(ghostIndex))
            else:
                value += self.ExpectedValue(gameState.getNextState(ghostIndex,Action),currentDepth,ghostIndex+1) / len(gameState.getLegalActions(ghostIndex))
        return value

    def MaxValue(self, gameState, currentDepth):
        if self.TerminalTest(gameState) or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        value = -float('inf')
        for Action in gameState.getLegalActions(0):
            value = max(value,self.ExpectedValue(gameState.getNextState(0,Action),currentDepth,1))
        return value

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foods = newFood.asList()
    evaluation = 0
    for i in range(len(newGhostStates)):
        ghostPosition = currentGameState.getGhostPosition(i + 1)
        distance = util.manhattanDistance(newPos, ghostPosition)
        if distance < 2:
            evaluation -= 2
        if distance <= newScaredTimes[i]:
            evaluation += distance
        if currentGameState.hasWall(newPos[0], newPos[1]):
            evaluation -= 2
        if newPos in foods:
            evaluation += 1
        foodDistance = []
        for food in foods:
            foodDistance.append(util.manhattanDistance(food, newPos))
        if foodDistance:
            evaluation -= 0.1 * min(foodDistance)
        if newPos in currentGameState.getCapsules() and distance >= 2:
            evaluation += 2
    return currentGameState.getScore() + evaluation


# Abbreviation
better = betterEvaluationFunction
