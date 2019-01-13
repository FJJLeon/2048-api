# 2048-api
A 2048 game api for training supervised learning (imitation learning) or reinforcement learning agents

# For TA 
* if you want to evaluate the model, run blow command in the path of ```/2048-api```
```
    python evaluate.py
```
note: the models should be in the path of ```/2048-api/model```
* if you want to test train the model
1. first modify the import part of **myAgent2.py** as below
```
    # test use
    #from game2048.agents import Agent, ExpectiMaxAgent
    #from game2048.game import Game
    # train use
    from agents import Agent, ExpectiMaxAgent
    from game import Game
```
note: if there exists models in the path ```/2048-api/model```, set **firstTime** to **False** in instantiation myOwnAgent in the last part of **myAgent2.py**, or set to **True**

2. then run blow command in the path of ```/2048-api```
```
    python game2048/myAgent2.py
```

# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
    * [`myAgent2.py`](game2048/myAgent2.py): my weak agent.
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.

# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask

# To define your own agents
```python
from game2048.agents import Agent

class YourOwnAgent(Agent):

    def step(self):
        '''To define the agent's 1-step behavior given the `game`.
        You can find more instance in [`agents.py`](game2048/agents.py).
        
        :return direction: 0: left, 1: down, 2: right, 3: up
        '''
        direction = some_function(self.game)
        return direction

```

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

# To run the web app
```bash
python webapp.py
```
![demo](preview2048.gif)

# LICENSE
The code is under Apache-2.0 License.

# For EE369 students from SJTU only
Please read [here](EE369.md).