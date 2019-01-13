from game2048.game import Game
from game2048.displays import Display

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def single_run(size, score_to_win, testAgent, **kwargs):
    # reboot game
    testAgent.resetGame()
    sumTime = testAgent.play(verbose=True)
    return game.score, sumTime


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50

    '''====================
    Use your own agent here.'''
    from game2048.agents import ExpectiMaxAgent as TestAgent
    '''===================='''
    from game2048.myAgent2 import myOwnAgent
    #myag = myOwnAgent()
    #myOwnAgent.load_model(['level1.h5','level2.h5','level3.h5','level4.h5'])

    game = Game(GAME_SIZE, SCORE_TO_WIN)
    agent = myOwnAgent(game=game, 
                maxPower=16,
                modelLevel=[256, 512, 900],
                display=Display())
    agent.load_model(['level1.h5','level2.h5','level3.h5','level4.h5'])

    scores = []
    sumTimes = []
    for _ in range(N_TESTS):
        score, sumTime = single_run(GAME_SIZE, SCORE_TO_WIN, testAgent=agent)
        scores.append(score)
        sumTimes.extend(sumTime)

    print("---==Stability==---")
    print("score:")
    npscores = np.array(scores)
    scoreCount = pd.value_counts(npscores, sort=True).sort_index()
    print(scoreCount)

    #print("step:")
    npsumtimes = np.array(sumTimes)
    stepCount = pd.value_counts(npsumtimes, sort=True).sort_index()
    #print(stepCount)

    if (False):
        x = scoreCount.index
        y = scoreCount.values
        scoreCount.plot(kind="bar")
        plt.xticks(rotation=360)
        #plt.yticks(np.arange(0,max(y)+1,1))
        plt.xlabel("score")
        plt.ylabel("score frequency", )
        plt.title("Score Frequency Histogram")
        plt.show()
        stepCount.plot(kind="bar")
        plt.xticks(rotation=90)
        plt.xlabel("step cost (ms)")
        plt.ylabel("frequency of step cost")
        plt.title("Step Cost Frequency Histogram")
        plt.show()
        
    print("Average time @%d steps: %d ms" % (len(sumTimes), sum(sumTimes)/len(sumTimes)))
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
    
