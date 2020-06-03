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
    '''Neural Network used to predict the movement for a snake'''
    def __init__(self):
        super(Snakebrain, self).__init__()
        self.model = Sequential()
        self.model.add(Dense(20, input_dim=104, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(4, activation= 'relu'))
        self.snake_name = " "
        self.fitness_score = 0.0
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

class Genetic_alg:
    def __init__(self, init_pop_size, generation_size, mutation_rate, percent_shift_mutation):
        self.fitness_rank = collections.deque()
        self.snake_brain_dict = {}
        self.startingpop = self.Initial_pop(init_pop_size)
        self.pop_size = init_pop_size
        self.current_gen = self.startingpop

        # variables for choose survivors funct.
        self.top_survival_rate = .2
        self.rand_survival_rate = .1

        #variables for breed funct
        self.generation_size = generation_size

        #variables for mutate funct
        self.mutation_rate = mutation_rate
        self.percent_shift_mutation = percent_shift_mutation
    def choose_survivors(self):
        '''chooses the survivors for each generation by selecting the top peformers and a few random survivors 
        top_survival_rate: decimal percentage indicating the percentage of top performers to survive
        rand_survival_rate: percentage of random snakes to survive
        snakes: list of snakes from previous generation in order of best to worst
        num_snakes: number of snakes given to function
        survivors contains top performers in the front and random survivors at the end'''
        num_top_survivors = int(self.pop_size*self.top_survival_rate)
        num_rand_survivors = int(self.pop_size * self.rand_survival_rate)
        survivors = []
        snakes = self.fitness_rank
        try:
            for i in range(num_top_survivors):
                survivors.append(snakes.popleft())
            for i in range(num_rand_survivors):
                survivors.append(rand.choice(snakes))
        except IndexError:
            print("More survivors than snakes in generation")
        #print("Top Performer:", survivors[0][0].name) #prints the name of the top performing snakebrain
        print('Survivors:', survivors)
        self.current_gen = survivors

    def breed_generation(self):
        print("Breeding...")
        next_gen = []  #a list of snakes that heve been bred from the previous generation's snakes
        chance_to_breed = []
        for snake in self.current_gen:
            chance_to_breed.append(snake[0].fitness_score)
        for i in range(self.generation_size):
            child = None
            mother = rand.choices(self.current_gen, weights = chance_to_breed)[0] #chooses parents more likely to choose parents with a higher fitness score
            father = rand.choices(self.current_gen, weights = chance_to_breed)[0]
            child = breed(mother, father)
            child = mutate(child, .05, .5)
            child.snake_name = i
            next_gen.append(child)
        print("Next Generation:", next_gen)
        self.current_gen = next_gen

    def breed(mother, father):
        ''' merge two nueral networks into one by randomly choosing weights/baises from each parent NN to pass onto the child NN
        goes through weights and biases and flips a coin for each weight/baise to decide whether or not to inherit from mother or father'''
        #print(mother , father)   debugging tool for finding mother and father 
        child = Snakebrain()
        params = mother[0].get_weights()
        if mother != father:
            fatherparams = father[0].get_weights()
            for i, val in np.ndenumerate(params): #iterate through each layers weights/biases
                for i2, val2 in np.ndenumerate(val): #iterates through each weight/biase array 
                    if rand.randint(0,1) ==1:
                        params[i[0]][i2] = fatherparams[i[0]][i2] #change weight from mother's to father's
            child.set_weights(params)
            return(child)
        if mother == father:
            child.set_weights(params)
            return child

    def mutate(self, Snake_NN):
        '''randomly alter the weights/ biases of a snake NN'''
        mutation_chance = int(self.mutation_rate*100)
        params = Snake_NN.get_weights()
        for i, val in np.ndenumerate(params): #iterate through each layers weights/biases
                for i2, val2 in np.ndenumerate(val): #iterates through each weight/biase array 
                    if rand.randint(0,100) <= mutation_chance:
                        weight = params[i[0]][i2]
                        shift = weight*self.percent_shift_mutation
                        if rand.randint(0,1)==1:
                            weight += shift
                        else:
                            weight -= shift
                        params[i[0]][i2] = weight
        Snake_NN.set_weights(params)
        return Snake_NN

    def fitness(snake_model):
        steps = float(snake_model.num_steps)
        food = float(snake_model.points_earned)
        fitfunc = steps + (2**food + (food**2.1)*500) - ((food**1.2) * (.25*steps)**1.3)
        return fitfunc 

    def simulate_generation(self, step_limit):
        '''simulates a generation of snakes on the model
        list_of_snakes:list containing snakebrains
        step_limit: number of timesteps to take before killing a snake manually'''
        print("Simulating Generation...")
        fitness_rank = collections.deque()
        snake_brain_dict = {} 
        for i, snake in enumerate(self.current_gen):
            model = SnakeModel(10, 10)
            #snake.snake_name = i
            for i2 in range(step_limit):
                try:
                    model.direction = snake.prediction(model.Nump_input())
                    model.one_step()
                except GameOver:
                    pass
            snake.fitness_score = fitness(model)
            snake_brain_dict[i] = [snake, snake.fitness_score]
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
        self.fitness_rank = fitness_rank
        self.snake_brain_dict = snake_brain_dict

    def Initial_pop(self, Initial_Pop_Size):
        Gen1 = []
        for i in range(Initial_Pop_Size):
            brain = Snakebrain()
            brain.snake_name = i
            Gen1.append(brain)
        return Gen1
if __name__ == "__main__":
    Nature = Genetic_alg(10,10,.01,.5)
    Nature.simulate_generation(100)
    Nature.choose_survivors()
    Nature.breed_generation()
    Nature.simulate_generation(100)
    print(Nature.fitness_rank)
    