# test use
from game2048.agents import Agent, ExpectiMaxAgent
from game2048.game import Game
# train use
#from agents import Agent, ExpectiMaxAgent
#from game import Game
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Input, Concatenate, AveragePooling2D, Activation
from keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
import time

class myOwnAgent(Agent):
    def __init__(self, game, maxPower=16, modelLevel=[], firstTime=False, display=None, path=['level1.h5','level2.h5','level3.h5','level4.h5']):
        if game.size != 4:
            raise ValueError("`%s` can only work with game of `size` 4." % self.__class__.__name__)
        self.game = game
        self.display = display
        self.maxPower = maxPower
        self.path = path
        # init models 
        modelLevel.append(np.inf)
        modelLevel = np.array(modelLevel)
        self.modelTarget = modelLevel
        self.firstTime = firstTime
        if firstTime:
            self.modelList = list()
            for max_score in modelLevel:
                self.modelList.append(self.build_model(maxPower=maxPower))
        else:
            print("loading model...")
            self.load_model()
        # if run as train, save model
        if __name__ == "__main__":
            self.save_model(self.path)

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        nowTime = lambda:int(round(time.time()*1000))
        sumTime = []
        while (n_iter < max_iter) and (not self.game.end):
            start_time = nowTime()
            direction = self.step()
            end_time = nowTime()
            sumTime.append(end_time - start_time)

            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)
        return sumTime
        
        
    def step(self):
        # 0: left, 1: down, 2: right, 3: up
        # convert current board to format board
        currentBoard = self.game.board
        formatInput = np.expand_dims(board2input(currentBoard, self.maxPower), axis=0)
        # select corresponding model
        currentScore = self.game.score
        index = np.sum(currentScore > self.modelTarget)
        # make direction
        choice = self.modelList[index].predict(formatInput)[0]
        direction = np.where(np.max(choice) == choice)[0][0]
        return direction

    def resetGame(self):
        self.game.reset()

    def build_model(self, maxPower):
        # x = Input((4,4,16))
        boardinputs = Input(shape=(self.game.size, self.game.size, maxPower)) 
        FILTERS = 128
        # conv layer (4*1,1*4,2*2,3*3,4*4)
        conv41 = Conv2D(filters=FILTERS,kernel_size=(4, 1),padding='same',kernel_initializer='he_uniform')(boardinputs)
        conv14 = Conv2D(filters=FILTERS,kernel_size=(1, 4),padding='same',kernel_initializer='he_uniform')(boardinputs)
        conv22 = Conv2D(filters=FILTERS,kernel_size=(2, 2),padding='same',kernel_initializer='he_uniform')(boardinputs)
        conv33 = Conv2D(filters=FILTERS,kernel_size=(3, 3),padding='same',kernel_initializer='he_uniform')(boardinputs)
        conv44 = Conv2D(filters=FILTERS,kernel_size=(4, 4),padding='same',kernel_initializer='he_uniform')(boardinputs)
        hidden = keras.layers.add([conv41, conv14, conv22, conv33, conv44])
        hidden = BatchNormalization(axis=-1)(hidden)
        #y = Activation('relu')(hidden)
        y = LeakyReLU(alpha=0.2)(hidden)

        # Flatten Dense
        y = Flatten()(y)
        for width in [512, 128]:
            y = Dense(width, kernel_initializer='he_uniform')(y)
            y = BatchNormalization()(y)
            y = LeakyReLU(alpha=0.2)(y)
        # Output
        outputs = Dense(4, activation='softmax')(y)
        model = Model(boardinputs, outputs)
        model.summary()
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        return model


    def train(self, expert, train_batch_size=32, batches_round=1000, epoch_per_train=10, max_loop=30):
        for i, max_score in enumerate(self.modelTarget):
            # find the model target score interval
            if i == 0:
                min_score = 0
            else:
                min_score = self.modelTarget[i - 1]
            finish = False
            print('----===== Training network %d score [%d, %d] =====----' % (i, min_score, max_score))

            # data generator
            # the weak_agent should be last one or self
            data_generator = data_gene_func(self, min_score, max_score, train_batch_size)
            # train epoch max 30, or fail          
            for out_loop in range(max_loop):
                print('---== Network %d Epoch %d in %d ==---' % (i, out_loop, max_loop))
                start_time = time.time()

                if (out_loop % 3 == 1):
                    self.save_model(self.path)

                # create data for 32 board * 1000 batch
                train_x = list()
                print('creating board...')
                for j in range(1,batches_round+1):
                    if j % 200 == 0:
                        print('%d batches created' % (j))
                    train_x.extend(data_generator.__next__())

                print('labeling...')
                train_y = list()
                for j, index in enumerate(train_x):
                    # label the board
                    expert.game.board = np.array(index)
                    train_y.append(expert.step())
                    train_x[j] = board2input(train_x[j], maxPower=self.maxPower)
                # one-hot convert to np.array
                train_x = np.array(train_x)
                train_y = to_categorical(train_y, num_classes=4)

                print('training...')
                h = self.modelList[i].fit(
                    train_x,
                    train_y,
                    batch_size=train_batch_size,
                    epochs=epoch_per_train,
                    verbose=1,
                    validation_split=0.1)

                # play 10 round game
                round_time = 10
                valid = 0.9
                score_sum = 0

                for _ in range(round_time):
                    self.game.reset()
                    while not self.game.end:
                        direction = self.step()
                        self.game.move(direction)
                    score_sum += self.game.score
                    
                # judge whether valid
                print('---== average score %.1f, expect %.1f ==---' % (score_sum / round_time, max_score * valid))
                if score_sum >= max_score * valid * round_time:
                    print('loop %d ,model %d finish' % (out_loop, i))
                    finish = True
                    break
                else:
                    print('loop %d ,model %d continue' % (out_loop, i))

                end_time = time.time()
                print("---== Epoch %d cost %d s ==---" % (out_loop, end_time-start_time))

            if finish:
                self.save_model(self.path)
                continue
            else:
                print('!!!!!model %d training failed!!!!!' % i)
                break

    def load_model(self, path=['level1.h5','level2.h5','level3.h5','level4.h5']):
        modelList = list()
        for p in path:
            modelList.append(keras.models.load_model('model/'+p))
        # set model to saved model
        self.modelList = modelList
        return modelList

    def save_model(self, path=['level1.h5','level2.h5','level3.h5','level4.h5']):
        for i in range(len(self.modelList)):
            keras.models.save_model(self.modelList[i], 'model/'+path[i])

def board2input(board, maxPower):
    # convert board to format input using one-hot
    size = len(board) # here 4
    formatInput = np.zeros((size, size, maxPower))
    xx, yy = np.meshgrid(range(size), range(size))
    xx, yy = xx.flatten(), yy.flatten()
    for (x, y) in zip(xx, yy):
        if board[x, y] != 0:
            # here may be bug, pos overflow
            pos = int(np.log2(board[x, y]) - 1)
            formatInput[x, y, pos] = 1
    # print(formatInput.shape)
    return formatInput

def data_gene_func(weak_agent, board_min=2, board_max=np.inf, batch_size=32):
    # Generator
    # using weak_agent give board data where score > board_min
    buffer = list()
    while True:
        # reset weak agent game
        weak_agent.game.reset()
        # record board between min and max score
        # maybe cannot gennerate min score, lower it by mul 0.6
        while (not weak_agent.game.end) and (weak_agent.game.score <= board_max):
            if weak_agent.game.score > board_min*0.6:
                buffer.append(weak_agent.game.board)
            direction = weak_agent.step()
            weak_agent.game.move(direction)
        # return a batch data
        while len(buffer) > batch_size:
            yield (buffer[:batch_size])
            buffer = buffer[batch_size:]

if __name__ == "__main__":
    tmpGame = Game(size=4, enable_rewrite_board=True)
    expertAgent = ExpectiMaxAgent(tmpGame)

    myAgent = myOwnAgent(
        game=tmpGame,
        maxPower=16, 
        modelLevel=[256, 512, 900],
        firstTime=False)

    myAgent.train(
        expertAgent,
        train_batch_size=32,
        batches_round=1000,
        epoch_per_train=10,
        max_loop=30)

    myAgent.save_model()