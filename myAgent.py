from game2048.agents import Agent
import numpy as np

class myOwnAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)

        #from .expectimax import board_to_move
        def hh(hhh):
            return np.random.randint(0, 4)
        self.search_func = hh

    def step(self):
        '''To define the agent's 1-step behavior given the `game`.
        You can find more instance in [`agents.py`](game2048/agents.py).
        
        :return direction: 0: left, 1: down, 2: right, 3: up
        '''
        
        direction = self.search_func(self.game.board)
        return direction