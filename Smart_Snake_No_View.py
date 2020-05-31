import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras import models
from tensorflow.keras.models import Sequential 
import tkinter as tk
from tkinter.font import Font
from enum import IntEnum
import unittest
import random as rand
import collections
from datetime import datetime
import math
import collections

class GameOver(Exception):
    pass

class SnakeModel:
    """ The model """
    def __init__(self, num_rows, num_cols):
        """ initialize the snake model """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_steps = 0
        self.points_earned = 0
        self.food_location = ()
        self.open_cells = []
        self.snake = collections.deque()
        self.state = [[CellState.Nothing for c in range(self.num_cols)] 
                        for r in range(self.num_rows)]
        self.NN_input = None
        
        #random food and snake start positions
        self.col = rand.randrange(0,self.num_cols)
        self.row = rand.randrange(0,self.num_rows)
        self.state = self.make_snake(self.row, self.col, self.state)
        self.snake.append((self.row, self.col))
        for c in range(self.num_cols):
            for r in range(self.num_rows):
                if r != self.row and c != self.col:
                    self.open_cells.append((r,c))
        self.state = self.make_food(self.state)
        
        #choose direction
        self.direction = None
        self.first_direction()
        #NN input 
        self.NN_input = self.Nump_input()

    def first_direction(self):
        '''Chooses an initial direction based on starting position of snake'''
        distance_edge = 0
        if self.col > distance_edge:
            self.direction = DirectionState.left
            distance_edge = self.col
        if (self.num_cols-self.col) > distance_edge:
            self.direction = DirectionState.right
            distance_edge = self.num_cols-self.col
        if self.row > distance_edge:
            self.direction = DirectionState.up
            distance_edge = self.row
        if (self.num_rows-self.row) > distance_edge:
            self.direction = DirectionState.down
            distance_edge = self.num_rows-self.row

    def make_snake(self, row, col, state):
        "make square into snake"
        state[row][col] = CellState.Snake
        if row != self.row and col != self.col:
            if (row, col) in self.open_cells:
                self.open_cells.remove((row, col))
        return state
    
    def make_food(self, state):
        "make square into food"
        randcell = rand.randrange(len(self.open_cells))
        row = self.open_cells[randcell][0]
        col = self.open_cells[randcell][1]
        self.food_location = (row, col)
        state[row][col] = CellState.Food
        return state

    def make_nothing(self, row, col, state):
        "make square into nothing"
        state[row][col] = CellState.Nothing
        self.open_cells.append((row, col))
        return state

    def reset(self):
        """ Resets all cells to nothing"""
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                self.make_nothing(r, c, self.state)

    def one_step(self):
        """ Simulates one time step of simulation """
        # Make array for state in next timestep
        next_state = [[CellState.Nothing for c in range(self.num_cols)] 
                        for r in range(self.num_rows)]
        snake_head = self.snake[-1]
        if self.direction == DirectionState.up and snake_head[0] != 0:
            r = -1
            c = 0
        elif self.direction == DirectionState.down and snake_head[0] != self.num_rows-1:
            r = 1
            c = 0
        elif self.direction == DirectionState.left and snake_head[1] != 0:
            r = 0
            c = -1
        elif self.direction == DirectionState.right and snake_head[1] != self.num_cols-1:
            r = 0
            c = 1  
        else:
            raise GameOver  #you lose

        if self.state[snake_head[0]+r][snake_head[1]+c] == CellState.Snake:
            raise GameOver         #end game

        if self.state[snake_head[0]+r][snake_head[1]+c] == CellState.Food:
            next_state = self.make_snake(snake_head[0]+r, snake_head[1]+c, next_state)
            self.snake.append((snake_head[0]+r, snake_head[1]+c)) 
            next_state = self.make_food(next_state)
            self.points_earned +=1

        if self.state[snake_head[0]+r][snake_head[1]+c] == CellState.Nothing:
            next_state = self.make_snake(snake_head[0]+r, snake_head[1]+c, next_state)
            next_state = self.make_nothing(self.snake[0][0], self.snake[0][1], next_state)
            self.snake.append((snake_head[0]+r, snake_head[1]+c))
            a = self.snake.popleft()
            next_state[self.food_location[0]][self.food_location[1]] = CellState.Food

        for i in self.snake:        #updates snake squares into next state
            next_state = self.make_snake(i[0], i[1], next_state)
        self.num_steps += 1
        self.NN_input = self.Nump_input()
        self.state = next_state
    
    def Nump_input(self):
        a = np.asarray(self.state, dtype='float32')
        a = a.flatten(order ='C')
        direc = np.array([0., 0., 0., 0.])
        if self.direction == DirectionState.up:
            direc[0]=1
        if self.direction == DirectionState.down:
            direc[1]=1
        if self.direction == DirectionState.left:
            direc[2]=1
        if self.direction == DirectionState.right:
            direc[3]=1
        a = np.append(a,direc)
        a = np.reshape(a, (1,104), order = 'C')
        return a

class Snakebrain(tf.keras.Model):
    def __init__(self):
        super(Snakebrain, self).__init__()
        self.model = Sequential()
        self.model.add(Dense(20, input_dim=104, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(4, activation= 'relu'))
        self.snake_name = " "
    def prediction(self, NN_input):
        max = 0
        direction = None
        a = self.model.predict(NN_input)
        if a[0][0] >= max:
            max = a[0][0]
            direction = DirectionState.up
        if a[0][1] >= max:
            max = a[0][1]
            direction = DirectionState.down
        if a[0][2] >= max:
            max = a[0][2]
            direction = DirectionState.left
        if a[0][3] >= max:
            max = a[0][3]
            direction = DirectionState.right
        return direction
    
    def __str__(self):
        """
        Returns string representation of the snake brain object.
        Handy for debugging.
        """
        return 'Snakebrain(%s)' % str(self.snake_name)

    def __repr__(self):
        """
        Returns string representation of the snake brain object.
        """
        return '%s' % str(self.snake_name)

class CellState(IntEnum):
    """ 
    Use IntEnum so that the test code below can
    set cell states using 0's 1's and 2's
    """
    Nothing = 0
    Snake = -1
    Food = 1

class DirectionState(IntEnum):
    up = 0
    down = 1 
    right = 2
    left = 3

class GameState(IntEnum):
    Initial = 0 
    Playing = 1
    Paused = 2
    Ended = 3

def choose_survivors(snakes, top_survival_rate, rand_survival_rate, num_snakes):
    '''chooses the survivors for each generation by selecting the top peformers and a few random survivors 
    top_survival_rate: decimal percentage indicating the percentage of top performers to survive
    rand_survival_rate: percentage of random snakes to survive
    snakes: list of snakes from previous generation in order of best to worst
    num_snakes: number of snakes given to function
    survivors contains top performers in the front and random survivors at the end'''
    num_top_survivors = int(num_snakes*top_survival_rate)
    num_rand_survivors = int(num_snakes * rand_survival_rate)
    survivors = collections.deque()
    for i in range(num_top_survivors):
        survivors.append(snakes.popleft())
    for i in range(num_rand_survivors):
        survivors.append(rand.choice(snakes))
    #print(survivors[0][0].name) #prints the name of the top performing snakebrain
    return survivors

def breed_generation(snakes, num_snakes_out):
    for i in range(num_snakes_out):
        mother = rand.choice(snakes)
        father = rand.choice(snakes)
        breed(mother, father)

def breed(mother, father):
    ''' merge two nueral networks into one by randomly choosing weights/baises from each parent NN'''
    if mother != father or mother == father:
        print(mother , father)
        params = mother[0].get_weights()
        fatherparams = father[0].get_weights()
        print(params[0][0,0])
        for layer in params:
            print('yooooooooooooooo')
            #print(layer)
            for weights in layer: 
                try:
                    r = 0
                    for row in weights:
                        c = 0
                        for col in weights:
                        #print(weight)
                            if True:
                                params[0][r, c] =0
                                c+=1
                        r+=1
                except TypeError:
                    pass
        print(params)


def mutate(Snake_NN, mutation_rate):
    '''randomly alter the weights/ biases of a snake NN'''
    pass

if __name__ == "__main__":
    #play of the game
    fitness_rank = collections.deque()
    snake_brain_dict = {}
    i = 0
    while(i<10):
        brain = Snakebrain()
        brain.snake_name = i
        model = SnakeModel(10, 10)
        try:
            while model.num_steps < 100:
                model.direction = brain.prediction(model.Nump_input())
                model.one_step()   
        except GameOver:
            pass
        snake_brain_dict[i] = [brain, model.num_steps]
        if i == 0:
            fitness_rank.appendleft(snake_brain_dict[i]) 
        else:
            b=0
            while True:
                try:
                    if fitness_rank[b][1] < snake_brain_dict[i][1]:
                        fitness_rank.insert(b, snake_brain_dict[i])
                        b+=1
                        break
                    b+=1
                except IndexError:
                    fitness_rank.append(snake_brain_dict[i])
                    break
        i += 1
    print(fitness_rank)
    aaa =  choose_survivors(fitness_rank,.2, .1, 10)
    print(aaa)
    breed_generation(aaa, 1)