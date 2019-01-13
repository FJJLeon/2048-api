#from game2048.agents import Agent
#from game2048.game import Game
from agents import Agent
from game import Game

import numpy as np
from  keras.layers import Dense, Conv2D, BatchNormalization,Flatten,Input,Concatenate,AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU

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

    def addBlock(self, inputs, num_filters):
        conv41 = Conv2D(num_filters, kernel_size=(2,2), kernel_initializer='he_uniform')(inputs)                                                                