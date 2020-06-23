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
class SnakeView:
    def __init__(self, num_rows, num_cols):
        """ Initialize view of the game """
       # Constants
        self.cell_size = 50
        self.control_frame_height = 100
        self.score_frame_width = 200

        # Size of grid
        self.num_rows = num_rows
        self.num_cols = num_cols

        # Create window
        self.window = tk.Tk()
        self.window.title("Snake Genetic Algorithm")
        
        self.grid_frame = tk.Frame(self.window, height = num_rows * self.cell_size,
                                width = num_cols * self.cell_size)
        self.grid_frame.grid(row = 1, column = 1) # use grid layout manager
        self.cells = self.add_cells()

        self.control_frame = tk.Frame(self.window, width = 800, 
                                height = self.control_frame_height)
        self.control_frame.grid(row = 2, column = 1, columnspan = 2) # use grid layout manager
        self.control_frame.grid_propagate(False)
        (self.start_button, self.reset_button, self.quit_button) = self.add_control()
    def schedule_next_step(self, step_time_millis, step_handler):
        """ schedule next step of the simulation """
        self.start_timer_object = self.window.after(step_time_millis, step_handler)

    def set_start_handler(self, handler):
        """ set handler for clicking on start button to the function handler """
        self.start_button.configure(command = handler)

    def set_reset_handler(self, handler):
        """ set handler for clicking on reset button to the function handler """
        self.reset_button.configure(command = handler)

    def set_quit_handler(self, handler):
        """ set handler for clicking on quit button to the function handler """
        self.quit_button.configure(command = handler)
    
    def make_snake_body(self, row, column):
        """ Make cell in row, column into snake """
        self.cells[row][column]['bg'] = 'blue'

    def make_snake_head(self, row, column):
        """ Make cell in row, column into snake head """
        self.cells[row][column]['bg'] = 'black'

    def make_nothing(self, row, column):
        """ Make cell in row, column nothing"""
        self.cells[row][column]['bg'] = 'white'

    def make_food(self, row, column):
        """ Make cell in row, column food """
        self.cells[row][column]['bg'] = 'red'

    def add_cells(self):
        """ Add cells to the grid frame """
        cells = []
        for r in range(self.num_rows):
            row = []
            for c in range(self.num_cols):
                frame = tk.Frame(self.grid_frame, width = self.cell_size, 
                           height = self.cell_size, borderwidth = 1, 
                           relief = "solid") 
                frame.grid(row = r, column = c) # use grid layout manager
                row.append(frame)
            cells.append(row)
        return cells
    def add_control(self):
        """Create control buttons and slider, and add them to the control frame"""
        start_button = tk.Button(self.control_frame, text="Start")
        start_button.grid(row=1, column=1, padx = 20)
        reset_button = tk.Button(self.control_frame, text="Reset")
        reset_button.grid(row=1, column=5, padx = 20)
        quit_button = tk.Button(self.control_frame, text="Quit")
        quit_button.grid(row=1, column=6, padx = 20)
        # Vertically center the controls in the control frame
        self.control_frame.grid_rowconfigure(1, weight = 1) 
        # Horizontally center the controls in the control frame
        self.control_frame.grid_columnconfigure(0, weight = 1) 
        self.control_frame.grid_columnconfigure(7, weight = 1) 
                                                            
        return (start_button, reset_button, quit_button)

    def update_view(self, cell_state):
        for row in range(self.num_rows):
                for col in range(self.num_cols):
                    if cell_state[row][col] == CellState.Snake:
                        self.make_snake_body(row, col)
                        #check for snake head
                    elif cell_state[row][col] == CellState.Food:
                        self.make_food(row, col)
                    else:
                        self.make_nothing(row, col)

    def reset(self):
        """reset all cells to nothing"""
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                self.make_nothing(r, c)
    def cancel_next_step(self):
        """ cancel the scheduled next step of simulation """
        self.window.after_cancel(self.start_timer_object)
    
class SnakeModel:
    """ The model """
    def __init__(self, num_rows, num_cols):
        """ initialize the snake model """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_steps = 0
        self.num_original_steps = 0
        self.points_earned = 0
        self.food_location = ()
        self.open_cells = []
        self.snake = collections.deque()
        self.state = [[CellState.Nothing for c in range(self.num_cols)] 
                        for r in range(self.num_rows)]
        self.NN_input = None
        self.food_state = self.state
        self.snake_state = self.state

        #random food and snake start positions
        self.col = rand.randrange(0,self.num_cols)
        self.row = rand.randrange(0,self.num_rows)
        self.state = self.make_snake(self.row, self.col, self.state)
        self.snake.append((self.row, self.col))

        self.prev_head = self.snake[0]
        self.prev2_head = self.prev_head
        for c in range(self.num_cols):
            for r in range(self.num_rows):
                if r != self.row and c != self.col:
                    self.open_cells.append((r,c))
        self.state = self.make_food(self.state)
        #choose direction
        self.direction = None
        self.first_direction()
        #NN input 
        self.snakeview_distance = 4
        self.distance_top_head = None
        self.distance_bottom_head = None
        self.distance_left_head = None
        self.distance_right_head = None
        self.distance_food_head = None
        self.angle_food_head = None
        self.distance_top_tail = None
        self.distance_bottom_tail = None
        self.distance_left_tail = None
        self.distance_right_tail = None
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
        self.snake_state[row][col] = CellState.Snake
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
        self.food_state[row][col] = 1
        return state

    def make_nothing(self, row, col, state):
        "make square into nothing"
        state[row][col] = CellState.Nothing
        self.food_state[row][col] = CellState.Nothing
        self.snake_state[row][col] = CellState.Nothing
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
        if snake_head != self.prev2_head:
            self.num_original_steps +=1
        self.prev2_head = self.prev_head 
        self.prev_head = snake_head
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
            self.num_steps = 0 #resets the model steps if food is found to prevent killing every snake after step limit is reached

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

    def snake_scan(self):
        head = self.snake[-1]
        forward = np.array([0., 0., 0., 0.])
        left = np.array([0., 0., 0., 0.])
        right = np.array([0., 0., 0., 0.])
        #print(self.direction, "Head:", head)

        if self.direction == DirectionState.up:
            for i in range(self.snakeview_distance):
                if head[0]-(i+1) >= 0: 
                    if self.state[head[0]-(i+1)][head[1]] == CellState.Snake:
                        forward[i] = 1
                if head[1]-(i+1) >= 0:
                    if self.state[head[0]][head[1]-(i+1)] == CellState.Snake:
                        left[i] = 1
                if head[1]+(i+1) <= self.num_cols-1:
                    if self.state[head[0]][head[1]+(i+1)] == CellState.Snake:
                        right[i] = 1

        if self.direction == DirectionState.down:
            for i in range(self.snakeview_distance):
                if head[0]+(i+1) <= self.num_rows-1: 
                    if self.state[head[0]+(i+1)][head[1]] == CellState.Snake:
                        forward[i] = 1
                if head[1]-(i+1) >= 0:
                    if self.state[head[0]][head[1]-(i+1)] == CellState.Snake:
                        right[i] = 1
                if head[1]+(i+1) <= self.num_cols-1:
                    if self.state[head[0]][head[1]+(i+1)] == CellState.Snake:
                        left[i] = 1
        
        if self.direction == DirectionState.left:
            for i in range(self.snakeview_distance):
                if head[1]-(i+1) >= 0: 
                    if self.state[head[0]][head[1]-(i+1)] == CellState.Snake:
                        forward[i] = 1
                if head[0]-(i+1) >= 0: 
                    if self.state[head[0]-(i+1)][head[1]] == CellState.Snake:
                        right[i] = 1
                if head[0]+(i+1) <= self.num_rows-1: 
                    if self.state[head[0]+(i+1)][head[1]] == CellState.Snake:
                        left[i] = 1

        if self.direction == DirectionState.right:
            for i in range(self.snakeview_distance):
                if head[1]+(i+1) <= self.num_cols-1:
                    if self.state[head[0]][head[1]+(i+1)] == CellState.Snake:
                        forward[i] = 1
                if head[0]-(i+1) >= 0: 
                    if self.state[head[0]-(i+1)][head[1]] == CellState.Snake:
                        left[i] = 1
                if head[0]+(i+1) <= self.num_rows-1: 
                    if self.state[head[0]+(i+1)][head[1]] == CellState.Snake:
                        right[i] = 1

        print(self.direction, forward, left, right)
        return [forward, left, right]

    def Nump_input(self):
        '''sets snake vision and inputs into a  numpy array'''
        head = self.snake[-1]
        tail = self.snake[0]
        self.distance_top_head = head[0]
        self.distance_top_tail= tail[0]
        self.distance_left_head = head[1]
        self.distance_left_tail = tail[1]
        self.distance_bottom_head = self.num_rows - head[0]
        self.distance_bottom_tail = self.num_rows - tail[0]
        self.distance_right_head = self.num_cols - head[1]
        self.distance_right_tail = self.num_cols - tail[1]
        self.distance_food_head = math.sqrt(((self.food_location[0]-head[0])**2) +((self.food_location[1] - head[1])**2))
        
        if head == tail:
            self.distance_top_tail = 0
            self.distance_bottom_tail = 0
            self.distance_left_tail = 0
            self.distance_right_tail = 0

        #calculates food angle from head
        if self.food_location[0] < head[0] and self.food_location[1] >= head[1]: #checks if food is in first quadrant with respect to head
            if self.food_location[1] == head[1]:
                self.angle_food_head = math.pi/2
            else:
                self.angle_food_head = math.atan((head[0]-self.food_location[0])/(self.food_location[1]-head[1]))
        elif self.food_location[0] <= head[0] and self.food_location[1] < head[1]: #checks if food is in second quadrant with respect to head
            self.angle_food_head = math.pi + math.atan((head[0]-self.food_location[0])/(self.food_location[1]-head[1]))
        elif self.food_location[0] >= head[0] and self.food_location[1] < head[1]: #checks if food is in third quadrant with respect to head
            self.angle_food_head = -math.pi - math.atan((head[0]-self.food_location[0])/(head[1]-self.food_location[1]))
        elif self.food_location[0] > head[0] and self.food_location[1] >= head[1]: #checks if food is in fourth quadrant with respect to head
            if self.food_location[1] == head[1]:
                self.angle_food_head = -math.pi / 2
            else:
                self.angle_food_head = - math.atan((head[0]-self.food_location[0])/(head[1]-self.food_location[1]))
        elif self.food_location[0] == head[0] and self.food_location[1] > head[1]:
            self.angle_food_head = 0
        else:
            print("could not calculate food angle")
            print("head:", head)
            print("food:", self.food_location)
        nump = np.array([self.distance_top_head,self.distance_top_tail, self.distance_left_head, self.distance_left_tail, self.distance_bottom_head, self.distance_bottom_tail, 
                        self.distance_right_head, self.distance_right_tail, self.distance_food_head, self.angle_food_head,self.points_earned])
        direc = np.array([0., 0., 0., 0.]) #Direction neurons and
        scan = self.snake_scan()
        forward = scan[0]
        left = scan[1]
        right = scan[2]
        if self.direction == DirectionState.up:
            direc[0]=1
        if self.direction == DirectionState.down:
            direc[1]=1
        if self.direction == DirectionState.left:
            direc[2]=1
        if self.direction == DirectionState.right:
            direc[3]=1
        nump = np.append(nump, direc)
        nump = np.append(nump, forward)
        nump = np.append(nump, left)
        nump = np.append(nump, right)
        nump = np.reshape(nump, (1,27), order = 'C')
        return nump

class Snakebrain(tf.keras.Model):
    '''Neural Network used to predict the movement for a snake'''
    def __init__(self):
        super(Snakebrain, self).__init__()
        self.model = Sequential()
        self.model.add(Dense(10, input_dim=27, activation='relu', use_bias= True, bias_initializer='glorot_uniform'))
        self.model.add(Dense(10, activation='relu',use_bias= True, bias_initializer='glorot_uniform'))
        self.model.add(Dense(4, activation= 'relu',))
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
        self.top_survival_rate = .1
        self.rand_survival_rate = .02

        #variables for breed funct
        self.generation_size = generation_size

        #variables for mutate funct
        self.mutation_rate = mutation_rate
        self.percent_shift_mutation = percent_shift_mutation
        self.chance_to_make_weight_negative = .05 
    
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
            child = self.breed(mother, father)
            child = self.mutate(child)
            child.snake_name = i
            print("Mother: ", mother, "Father:", father, "Child:", child)
            next_gen.append(child)
        print("Next Generation:", next_gen)
        self.current_gen = next_gen

    def breed(self, mother, father):
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
        negative_chance = int(self.chance_to_make_weight_negative*100)
        params = Snake_NN.get_weights()
        for i, val in np.ndenumerate(params): #iterate through each layers weights/biases
                for i2, val2 in np.ndenumerate(val): #iterates through each weight/biase array 
                    if rand.randint(0,100) <= mutation_chance:
                        weight = params[i[0]][i2]
                        if rand.randint(0,100) <= negative_chance:
                            weight = -weight
                        self.percent_shift_mutation = rand.randint(0,10)/10
                        shift = weight*self.percent_shift_mutation
                        if rand.randint(0,1)==1:
                            weight += shift
                        else:
                            weight -= shift
                        params[i[0]][i2] = weight
        Snake_NN.set_weights(params)
        return Snake_NN

    def fitness(self, snake_model):
        steps = float(snake_model.num_steps)
        food = float(snake_model.points_earned)
        original_steps = float(snake_model.num_original_steps)
        fitfunc = round(original_steps + (2**food + (food**2.1)*500) - ((food**1.2) * (.25*original_steps)**1.3), 8)
        return fitfunc 

    def Initial_pop(self, Initial_Pop_Size):
        Gen1 = []
        for i in range(Initial_Pop_Size):
            brain = Snakebrain()
            brain.snake_name = i
            Gen1.append(brain)
        return Gen1
class controller:
    def __init__(self):
        """ Initializes the snake game """
        #define parameters
        self.NUM_ROWS = 10
        self.NUM_COLS = 10
        self.model = None
        self.GameState = GameState.Initial
        self.step_time_millis = 700
        self.init_pop_size = 100
        self.generation_size = 100
        self.step_limit = 100
        self.genetic = Genetic_alg(self.init_pop_size, self.generation_size, .1, .2)
        self.brain_index = 0
        self.view = SnakeView(self.NUM_ROWS, self.NUM_COLS)
        self.model = SnakeModel(self.NUM_ROWS, self.NUM_COLS)
        self.generation_num = 0

        #connect buttons to functions 
        self.view.set_start_handler(self.start_handler)
        self.view.set_reset_handler(self.reset_handler)
        self.view.set_quit_handler(self.quit_handler)
        self.view.window.mainloop()
    def start_handler(self):
        if self.GameState != GameState.Playing and self.GameState != GameState.Ended:
            self.GameState = GameState.Playing
            self.view.schedule_next_step(self.step_time_millis, 
                                        self.continue_simulation)
    def reset_handler(self):
        """ Reset simulation """
        self.view.reset()
        self.model = SnakeModel(self.NUM_ROWS, self.NUM_COLS)
        self.GameState = GameState.Initial
    def quit_handler(self):
        """ Quit life program """
        self.view.window.destroy()
    def continue_simulation(self):
        """ Perform another step of the simulation, and schedule the next step."""
        if self.GameState == GameState.Playing:
            try:
                self.simulate_snake_step(self.genetic.current_gen[self.brain_index])
                self.view.schedule_next_step(self.step_time_millis, self.continue_simulation)
            except GameOver:
                self.view.update_view(self.model.state)
                self.genetic.current_gen[self.brain_index].fitness_score = self.genetic.fitness(self.model)
                self.genetic.snake_brain_dict[self.brain_index] = [self.genetic.current_gen[self.brain_index], self.genetic.current_gen[self.brain_index].fitness_score]
                if self.brain_index == 0:
                    self.genetic.fitness_rank.append(self.genetic.snake_brain_dict[self.brain_index])
                if self.brain_index <= len(self.genetic.current_gen):
                    b=0
                    while True:
                        try:
                            if self.genetic.fitness_rank[b][1] < self.genetic.snake_brain_dict[self.brain_index][1]:
                                self.genetic.fitness_rank.insert(b, self.genetic.snake_brain_dict[self.brain_index])
                                b+=1
                                break
                            b+=1
                        except IndexError:
                            self.genetic.fitness_rank.append(self.genetic.snake_brain_dict[self.brain_index])
                            break 
                print("Generation:",self.generation_num, "Snake: ", self.brain_index, "Score: ", self.genetic.snake_brain_dict[self.brain_index][1] ,"Top Score:", self.genetic.fitness_rank[0][1], "Points Earned: ", self.model.points_earned)
                self.reset_handler()
                self.brain_index +=1
                if self.brain_index <= len(self.genetic.current_gen):
                    self.start_handler()
                if self.brain_index >= len(self.genetic.current_gen):
                    print("Generation:",self.generation_num, "Snake: ", self.brain_index, "Top Score: ", self.genetic.fitness_rank[0][1])
                    print("Fitness: ", self.genetic.fitness_rank)
                    self.reset_handler()
                    self.view.window.destroy()
                    self.view = SnakeView(self.NUM_ROWS, self.NUM_COLS)
                    self.genetic.choose_survivors()
                    self.genetic.breed_generation()
                    self.genetic.fitness_rank = collections.deque()
                    self.brain_index = 0
                    self.generation_num += 1
                    self.start_handler()
    def check_direction(self, prev_direction, direction_prediction):
        if direction_prediction == DirectionState.down and prev_direction != DirectionState.up:
            self.model.direction = direction_prediction
        if direction_prediction == DirectionState.up and prev_direction != DirectionState.down:
            self.model.direction = direction_prediction
        if direction_prediction == DirectionState.left and prev_direction != DirectionState.right:
            self.model.direction = direction_prediction
        if direction_prediction == DirectionState.right and prev_direction != DirectionState.left:
            self.model.direction = direction_prediction      
    def simulate_snake_step(self, NN):
        if self.model.num_steps > self.step_limit:
            print("Too Many steps")
            raise GameOver
        self.view.update_view(self.model.state)
        self.check_direction(self.model.direction, NN.prediction(self.model.Nump_input()))
        self.model.one_step() 
    
if __name__ == "__main__":
   b = controller()