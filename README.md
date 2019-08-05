# RL-Continuing-Tasks

Python implementation of three Reinforcement Learning continuing tasks.

![](example.jpg)

# Task Description

## Task 1

The first task is inspired by the queuing problem described by Sutton and Barto [1].The environment consists of a queue containing customers of uniformly distributedpriorities of 1, 2, 4 and 8.  These customers can be assigned to 10 servers.  The agentmust decide whether to assign the customer at the front of the queue to a server orreject the customer. Assigning a customer of prioritypto server results in a reward of p. If all servers are full and the agent accepts a customer, the customer is automaticallyrejected. We introduce a reward of -1 in this event in order to avoid policies where theagent always accepts customers.  At each time step, each server has a probability of 0.06 of becoming free. Finally, the queue remains full at all times. The state representation consists of a vector of size 11 containing the priority of thefirst 10 customers in the queue, as well as the number of free servers. The agent has two actions available: accept, reject. 

## Task 2

The second task is inspired by the factory simulation introduced by Mahadevan et. al [2].  The environment consists of a machine that can produce one of 5 products andstore them in buffers of respective sizes (30, 20, 15, 15 and 10).  Demands for eachproduct have different stochastic inter-arrival times and prices (9, 7, 16, 20 or 25).  Ademand arrival for a product results in a reward equal to the product price.  Failurescan  occur  randomly  at  any  point  and  force  a  repair  resulting  in  a  reward  of  -5000 and the interruption of all production.  Failures can be avoided by performing regularmaintenance. Performing maintenance results in a reward of -500 and stops productionfor a shorter time than a repair.  We refer to the first system described by Mahadevan et. al [2] for the details regarding the distributions of demand arrivals, failure arrivals,repairs times and maintenance times.  After the completion of production, the end of a repair,  or the end of maintenance,  the agent must decide to produce one of the 5 products or to perform maintenance. When production starts for a product, it may not stop until the buffer is full or a failure occurs.

## Task 3
For the last task, we implemented a simple game where cubes fall from the top of a 10x10 pixel screen and the agent must move left, right, or remain in ti’s current positionto collect the squares and obtain a reward. 
* Each row contains a white square with probability 0.5. The square is placeduniformly in one of the 10 columns.
* At each time step, the rows are moved down by one.
* The grey square on the bottom row represents the agent
*At each time step, the agent can move left or right by one pixel or stay in itscurrent  position.   If  the  agent  and  a  white  square  are  in  the  same  position,  areward of 1 is received.

As there are more white squares than the agent can collect, the agent must not only learn to collect the white squares but also learn which ones to collect to maximizefuture rewards. The state-representation consists of a 10x10 pixel matrix where 0 represents a blackpixel and 255 for a white pixel. An example video can be found [here](https://youtu.be/P1GFhcgVdV8).

[1] Richard S Sutton and Andrew G Barto.Reinforcement learning: An introduction.MIT press, 2018.  
[2] Sridhar Mahadevan,  Nicholas Marchalleck,  Tapas K Das,  and Abhijit Gosavi.Self-improving  factory  simulation  using  continuous-time  average-reward  rein-forcement learning.  In MACHINE LEARNING-INTERNATIONAL WORKSHOPTHEN CONFERENCE-, pages 202–210. MORGAN KAUFMANN PUBLISH-ERS, INC., 1997.

# Usage
The code below initialises the environement for task 3 and takes random steps indefinitely. 

(Note that visualization was only tested on MacOS 10.14.5 using python 3.6.8)
```python
import random
from task3 import task3

Env=Env=task3(render=False) #render=True to enable vizualisation

while True:
  action= random.randint(0,2)
  state, reward=  Env.step(action)
```
