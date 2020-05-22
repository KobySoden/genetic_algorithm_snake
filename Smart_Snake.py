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

class GameOver(Exception):
    pass
class snake:
    """ This is the controller """
    def __init__(self):
        """ Initializes the snake game """
        #define parameters
        self.NUM_ROWS = 10
        self.NUM_COLS = 10
        self.model = None
        self.mode = Mode.Norm
        self.GameState = GameState.Initial
        self.step_time_millis = 500

        # Create view
        self.view = SnakeView(self.NUM_ROWS, self.NUM_COLS) #initialize snakeview object
        #create model
        self.model = SnakeModel(self.NUM_ROWS, self.NUM_COLS)
        # Start
        self.view.set_start_handler(self.start_handler)
        # Reset 
        self.view.set_reset_handler(self.reset_handler)
        # Quit
        self.view.set_quit_handler(self.quit_handler)
        self.time_elapsed = 0
        
        # Start the simulation
        self.view.window.mainloop()

    def start_handler(self):
        if self.GameState != GameState.Playing and self.GameState != GameState.Ended:
            self.GameState = GameState.Playing
            self.view.schedule_next_step(self.step_time_millis, 
                                        self.continue_simulation)
        print("Start simulation")

    def reset_handler(self):
        """ Reset simulation """
        self.view.reset()
        self.model = SnakeModel(self.NUM_ROWS, self.NUM_COLS)
        self.GameState = GameState.Initial
        self.view.time_diff.set('Time: 00.00')
        self.time_elapsed = 0
        self.view.score_per_second.set('Points per second: 0.00')
        print("Reset simulation")

    def quit_handler(self):
        """ Quit life program """
        self.view.window.destroy()

    def continue_simulation(self):
        """ Perform another step of the simulation, and schedule the next step."""
        if self.GameState == GameState.Playing:
            self.one_step()
            self.view.schedule_next_step(self.step_time_millis, self.continue_simulation)
            self.time_elapsed += self.step_time_millis/1000
    
    def one_step(self):
        """ Simulate one step """
        try:
            # Update the model
            self.model.one_step()
            # Update the view
            for row in range(self.NUM_ROWS):
                for col in range(self.NUM_COLS):
                    #print(row, col)
                    if self.model.state[row][col] == CellState.Snake:
                        self.view.make_snake_body(row, col)
                        #check for snake head
                    elif self.model.state[row][col] == CellState.Food:
                        self.view.make_food(row, col)
                    else:
                        self.view.make_nothing(row, col)
            head = self.model.snake[-1]
            self.view.make_snake_head(head[0], head[1])
            self.view.points_earned.set('Points: ' + str(self.model.points_earned))
            if self.time_elapsed != 0:
                self.view.score_per_second.set('Points per second: ' + str(round(self.model.points_earned/self.time_elapsed, 2)))
            self.view.time_diff.set('Time: ' + str(round(self.time_elapsed, 2)) + 's')

        except GameOver:
            self.GameState = GameState.Ended
            self.view.game_over.set('Game Over')

class SnakeView:
    def __init__(self, num_rows, num_cols):
        """ Initialize view of the game """
       # Constants
        self.cell_size = 20
        self.control_frame_height = 100
        self.score_frame_width = 200

        # Size of grid
        self.num_rows = num_rows
        self.num_cols = num_cols

        # Create window
        self.window = tk.Tk()
        self.window.title("Greedy Snake")

        #string variables for time and points
        self.points_earned = tk.StringVar()
        self.points_earned.set("Points: 0 ")
        self.game_over = tk.StringVar()
        self.game_over.set(" ")
        self.time_diff = tk.StringVar()
        self.time_diff.set("Time: 00:00")
        self.current_time = tk.StringVar()
        self.score_per_second = tk.StringVar()
        self.score_per_second.set('Points per second:     ')
        
 
        # Create frame for grid of cells
        self.grid_frame = tk.Frame(self.window, height = num_rows * self.cell_size,
                                width = num_cols * self.cell_size)
        self.grid_frame.grid(row = 1, column = 1) # use grid layout manager
        self.cells = self.add_cells()

        # Create frame for controls
        self.control_frame = tk.Frame(self.window, width = 800, 
                                height = self.control_frame_height)
        self.control_frame.grid(row = 2, column = 1, columnspan = 2) # use grid layout manager
        self.control_frame.grid_propagate(False)
        (self.start_button, self.reset_button, self.quit_button) = self.add_control()

        #Create frame for score 
        self.score_frame = tk.Frame(self.window, width = self.score_frame_width, 
                                height = num_rows * self.cell_size)
        self.score_frame.grid(row = 1, column = 2) # use grid layout manager
        self.score_frame.grid_propagate(False)
        (self.score_label, self.points_frame, self.time_frame,
                self.points_per_second_frame, self.gameover_label) = self.add_score()


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

    def add_score(self):
        """Create labels and small frames and add them to the score frame"""
        score_label = tk.Label(self.score_frame, text = 'Score', font = ("Times", 18) )
        score_label.grid(row = 0, column = 1, pady = 30, padx = 25, sticky = 'N')
        points_frame = tk.Frame(self.score_frame, highlightbackground = 'black', highlightthickness = 1)
        points_frame.grid(row = 2, column = 1, columnspan = 5, sticky = 'N')
        points_frame_label = tk.Label(points_frame, textvariable = self.points_earned)
        points_frame_label.grid(row = 1, column = 1, sticky = 'W')
        time_frame = tk.Frame(self.score_frame, highlightbackground = 'black', highlightthickness = 1)
        time_frame.grid(row = 3, column = 1, pady = 30, columnspan = 3, sticky = 'N')
        time_frame_label = tk.Label(time_frame, textvariable = self.time_diff)
        time_frame_label.grid(row = 1, column = 1, sticky = 'W')
        points_per_second_frame = tk.Frame(self.score_frame, highlightbackground = 'black', highlightthickness = 1)
        points_per_second_frame.grid(row = 4, column = 1, columnspan = 3, sticky = 'N')
        points_per_second_frame_label = tk.Label(points_per_second_frame, textvariable = self.score_per_second)
        points_per_second_frame_label.grid(row = 1, column = 1, sticky = 'W')
        gameover_label = tk.Label(self.score_frame, textvariable = self.game_over, font = ("Times", 18) )
        gameover_label.grid(row = 5, column = 1, pady = 30, sticky = 'N')
        #horizontally center the labels and frames in the score frame
        self.score_frame.grid_columnconfigure(0, weight = 1) 
        self.score_frame.grid_columnconfigure(7, weight = 1) 

        return(score_label, points_frame, time_frame,
                points_per_second_frame, gameover_label)

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

    def reset(self):
        """reset all cells to nothing"""
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                self.make_nothing(r, c)
        self.game_over.set(" ")
        self.points_earned.set("Points: 0")

    def schedule_next_step(self, step_time_millis, step_handler):
        """ schedule next step of the simulation """
        self.start_timer_object = self.window.after(step_time_millis, step_handler)

    def cancel_next_step(self):
        """ cancel the scheduled next step of simulation """
        self.window.after_cancel(self.start_timer_object)
    def set_start_handler(self, handler):
        """ set handler for clicking on start button to the function handler """
        self.start_button.configure(command = handler)

    def set_reset_handler(self, handler):
        """ set handler for clicking on reset button to the function handler """
        self.reset_button.configure(command = handler)

    def set_quit_handler(self, handler):
        """ set handler for clicking on quit button to the function handler """
        self.quit_button.configure(command = handler)

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
        a = a.flatten(order ='F')
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
        return a
        
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

class Mode(IntEnum):
    Norm = 0
    Wrap = 1

class GameState(IntEnum):
    Initial = 0 
    Playing = 1
    Paused = 2
    Ended = 3
