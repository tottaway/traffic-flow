import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import copy

from tqdm import tqdm

# Simulation Variables
T = 15 * 60         # Number of seconds
PLAYBACK_RATE = 1  # Seconds between update in animation (NOTE: UPDATE every 10ms)
LANES = 1           # Number of lanes on the road
DIST = 5000         # Length of the road in meters
SPAWN_PROB = 0.35   # Probability a car enters a lane in a given second

# http://digitalassets.lib.berkeley.edu/math/ucb/text/math_s2_article-43.pdf
# Source for traffic speed distribution
# TODO: cross validate this method
MEAN_V_MAX = 22.5 # mean desired speed of cars in m/s
STD_V_MAX = 3  # standard distribution of car speed in m/s

MAX_ACCELERATION = 4 # in m/s^2
MIN_ACCELERATION = -8
TAU = 0.5 # Constant account for delay in velocity change
ALPHA = 50 # Constant describing caution of drivers

class Car():
    def __init__(self):
        self.position = 0
        self.lane = np.random.randint(LANES) + 1
        self.v_max = np.random.normal(loc=MEAN_V_MAX, scale=STD_V_MAX)
        self.velocity = self.v_max

    def headway(self, all_cars):
        cars_in_lane = [car for car in all_cars if car.lane == self.lane]
        cars_ahead = [car for car in cars_in_lane if car.position > self.position]
        if len(cars_ahead) == 0:
            # car has no one ahead
            headway = 10000 # aribitrary large number (should probably change)
        else:
            headway = min([(car.position-self.position) for car in cars_ahead])
        return headway
    
    def increment(self, all_cars):
        headway = self.headway(all_cars)
        target_speed = self.v_max * np.tanh(headway / ALPHA)
        change = (target_speed - self.velocity) / TAU
        change = change if change < MAX_ACCELERATION else MAX_ACCELERATION
        change = change if change > MIN_ACCELERATION else MIN_ACCELERATION
        self.velocity += change
        # self.velocity = target_speed
        self.position += self.velocity

def run(T):
    state = [[]]
    for t in tqdm(range(1, T)):
        prev_state = copy.deepcopy(state[t-1])
        curr_state = []

        for car in prev_state:
            car.increment(prev_state)
            if car.position < DIST-1:
                curr_state.append(car)
                
        
        spawn = SPAWN_PROB > np.random.rand()
        if spawn:
            curr_state.append(Car())
        
        state.append(curr_state)
    return state

def get_data(state):
    x, y = [], []
    for car in state:
        x.append(car.position)
        y.append(car.lane)
    return x, y
fig, ax = plt.subplots()
plt.title("Traffic")
plt.xlabel("X (Meters)")
plt.ylabel("Lane")
plt.xlim(0, DIST)
plt.ylim(0, LANES+1)

data = run(T)
x, y = [], []
scat, = ax.plot(x, y, marker="o", ls="", alpha=0.25)
def update(t):
    t = int(t * PLAYBACK_RATE)
    new_x, new_y = get_data(data[t])
    scat.set_data(new_x, new_y)
    return scat,

ani = FuncAnimation(fig, update, frames=int(T/PLAYBACK_RATE), interval=10)
plt.show()
