#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 09:28:33 2026

@author: gracetait
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import argparse
from scipy.optimize import curve_fit

class GameOfLife(object):
    
    def __init__(self, n, init_cond, p_alive):
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of sites in the lattice
        self.init_cond = init_cond # Choice of the initial lattice 
        self.lattice = None
        self.p_alive = p_alive
        
    def initialise(self):
        
        # Create a two-dimensional square lattice
        # Where 0 is dead and 1 is live
        # Create the initial lattice based on user input
        if self.init_cond == 'random':
        
            # Create a lattice according to a probability
            # Where p_alive is the probability a site is alive (1)
            # The default is p_alive = 0.5
            self.lattice = np.random.choice([0, 1], size = (self.n, self.n),
                                            p = [1 - self.p_alive, self.p_alive])
            
        elif self.init_cond == 'randomglider':
            
            # Create a lattice that has a glider
            self.lattice = np.zeros((self.n, self.n))
            
            # Choose a random site
            i = np.random.randint(self.n)
            j = np.random.randint(self.n)
            
            # Put in a glider
            self.lattice[i, j] = 1
            self.lattice[(i + 1) % self.n, (j + 1) % self.n] = 1
            self.lattice[(i + 2) % self.n, (j - 1) % self.n] = 1
            self.lattice[(i + 2) % self.n, j] = 1
            self.lattice[(i + 2) % self.n, (j + 1) % self.n] = 1
            
        elif self.init_cond == 'orderedglider':
            
            # Create a lattice that has a glider
            self.lattice = np.zeros((self.n, self.n))
            
            # Start the glider at the top left of the lattice
            i = 0
            j = 0
            
            # Put in a glider
            self.lattice[i, j] = 1
            self.lattice[(i + 1) % self.n, (j + 1) % self.n] = 1
            self.lattice[(i + 2) % self.n, (j - 1) % self.n] = 1
            self.lattice[(i + 2) % self.n, j] = 1
            self.lattice[(i + 2) % self.n, (j + 1) % self.n] = 1
            
        else:
            
            # Create a lattice
            self.lattice = np.zeros((self.n, self.n))
            
            # Choose a random site
            i = np.random.randint(self.n)
            j = np.random.randint(self.n)
            
            # Put in a blinker
            self.lattice[i, j] = 1
            self.lattice[i - 1, j] = 1
            self.lattice[i + 1, j] = 1
            
    def alive_or_dead(self, i, j):
        
        # Obtain the site 
        cell = self.lattice[i, j]
        
        # Count the number of live neighbours, accounting for periodic boundaries
        live_neighbours = int(self.lattice[(i - 1) % self.n, (j - 1) % self.n] + 
                              self.lattice[(i - 1) % self.n, j] + 
                              self.lattice[(i - 1) % self.n, (j + 1) % self.n] + 
                              self.lattice[i, (j - 1) % self.n] + 
                              self.lattice[i, (j +  1) % self.n] + 
                              self.lattice[(i + 1) % self.n, (j - 1) % self.n] + 
                              self.lattice[(i + 1) % self.n, j] + 
                              self.lattice[(i + 1) % self.n, (j + 1) % self.n])
        
        # If the cell is live ...
        if cell == 1: 
            
            # If live neighbours < 2 or > 3, the cell dies
            if live_neighbours < 2 or live_neighbours > 3:
                 return 0
                
             # Otherwise, it lives
            else:  
                 return 1
        
        # If the cell is dead ...
        else: 
            
            # live neighbours == 3, cell is alive
            if live_neighbours == 3:
                return 1
            
            # Otherwise, it stays dead
            else:
                return 0

    def total_alive_sites(self):
        
        return np.sum(self.lattice)
    
    def update_lattice(self):
        
        # Keep old lattice to check for updates
        lattice_new = np.zeros((self.n, self.n))
        
        # Iterate through the whole lattice
        # Update the new lattice at each iteration
        for i in range(self.n):
            
            for j in range(self.n):
            
                lattice_new[i, j] = self.alive_or_dead(i, j)
                
        # Once done, update the main lattice simultaneously
        self.lattice = lattice_new
                
        return self.lattice
        
    def center_of_mass(self):
        
        # Locate where the lattice has alive cells (1) this is the glider
        glider = np.argwhere(self.lattice == 1)
        
        # If there are no alive cells, the pattern died out
        if len(glider) == 0:
            print("The system is dead. RIP.")
            return None
        
        # Only continue if the glider is not at the boundaries
        if np.all(glider[:, 1] != 0) and np.all(glider[:, 0] != 0):
        
            # Calculate the mean of the coordinates
            x_cm = np.mean(glider[:, 0]) 
            y_cm = np.mean(glider[:, 1]) 
             
            # Calculate the hypotenuse
            r_cm = np.sqrt(x_cm**2 + y_cm**2)
            
            # Return the centre of mass
            return x_cm, y_cm, r_cm
        
        # If the glider is at the boundaries, do not calculate
        else:
            
            return np.nan, np.nan, np.nan  
    

class Simulation(object):
    
    def __init__(self, n, init_cond, steps):
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of sites in the square lattice
        self.init_cond = init_cond # Choice of the initial lattice 
        self.steps = steps # Choice of number of steps for the animation
        
    def animate(self, steps):
        
        # Initialise the lattice using the GameOfLife class
        game_of_life = GameOfLife(self.n, self.init_cond)
        game_of_life.initialise()
        
        # Define the figure and axes for the animation
        fig, ax = plt.subplots()
        
        # Define custom cmap
        red_black_cmap = ListedColormap(["#000000", "#FF0000"]) 
        
        # Initialise the image object
        # vmin/vmax ensure 0 is black and 1 is red consistently
        im = ax.imshow(game_of_life.lattice, cmap = red_black_cmap,
                       vmin = 0, vmax = 1)
        
        # Run the animation for the total number of steps
        for s in range(steps):
            
            # Update lattice
            game_of_life.update_lattice()
            
            # Update the animation 
            im.set_data(game_of_life.lattice)
            ax.set_title(f"Step: {s}")
            
            # Keep the image up while the script is running
            plt.pause(0.001)
            
        # Keep the final image open when the loop finishes
        plt.show()
        
    def equilibrium_check(self, active_sites, ind_cond = 10):
        
        # If the array if not long enough, return False
        if len(active_sites) < ind_cond:
            return False
        
        """
        # Obtain the last __ elements of the array
        last_elements = active_sites[-ind_cond:]
        
        # Check if the elements are all the same
        all_same = np.all(last_elements == last_elements[0])
        
        # Checks if the elements are oscillating
        is_oscillating = np.all(last_elements[2:] == last_elements[:-2])
        
        print(last_elements, all_same, is_oscillating)
        
        # If system has reached an active phase or absorbing state, return True
        if all_same == True or is_oscillating == True:
            return True
        
        # Otherwise, return False
        else:
            return False
        """
        
        # Obtain the last __ elements of the array
        last_elements = active_sites[-ind_cond:]
        
        # Calculate the mean and standard devation
        #mean = np.mean(last_elements)
        std = np.std(last_elements)
        
        # Define the threshold to be 1.5
        threshold = 1.5
        
        # If the std is below the threshold, the system is in equilbrium
        if std < threshold:
            return True
        
        # If not, it is not in equilibrium
        else:
            return False
        
    def equilibrium_measurements(self, filename, steps):
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders don't exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
        
        # Initialise the lattice using the GameOfLife class
        game_of_life = GameOfLife(self.n, self.init_cond)
        
        # Make empty lists to hold data points
        equilibriation_time = []
        no_active_sites = []
        
        # Iterate through simulation steps
        for s in range(steps):
            print(f"Simulating step = {s}/{steps}")
            
            # Initialise the lattice
            game_of_life.initialise()
            
            # Empty list to hold number of active sites
            active_sites = []
            
            # Set equilibrium as False at the beginning
            equilibrium = False
            
            # Set time to zero
            time = 0
            
            # Continue until equilibrium has been reached
            while equilibrium == False:
                
                # Update lattice
                game_of_life.update_lattice()
                
                active_sites.append(game_of_life.total_alive_sites())
                
                # Update time
                time += 1

                # Check if equilibrium has been reached
                equilibrium = self.equilibrium_check(active_sites) # Can change ind_cond
                
            # Append values to the lists
            equilibriation_time.append(time)
            no_active_sites.append(active_sites)
            
        # Open in "a" (append) or "w" (overwrite) mode
        # Write the values into the specified file
        with open(file_path, "a") as f:
            for i in range(len(equilibriation_time)):
                
                # Get the time it took
                t = equilibriation_time[i]
                
                # Get the final active site count (the last value in that sub-list)
                final_count = no_active_sites[i][-1]
                
                f.write(f"{t},{final_count}\n")
                
    def glider_measurements(self, steps, filename):
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders don't exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
        
        # Initialise the lattice using the GameOfLife class
        game_of_life = GameOfLife(self.n, self.init_cond)
        game_of_life.initialise()
        
        # Create empty lists to hold datapoints
        x = []
        y = []
        r = []
        time = []
        
        # Run the measurements for the total number of steps
        for s in range(steps):
            #print(f"Simulating step = {s}/{steps}")
            
            # Update lattice
            game_of_life.update_lattice()
            
            # Calculate glider center of mass
            x_cm, y_cm, r_cm = game_of_life.center_of_mass()
            
            # If the glider is crossing a boundary, do not append to list
            if np.isnan(x_cm) or np.isnan(y_cm) or np.isnan(r_cm):
                continue
            
            else:
                
                x.append(x_cm)
                y.append(y_cm)
                r.append(r_cm)
                time.append(s)
                
        # Open in "a" (append) or "w" (overwrite) mode
        # Write the values into the specified file
        with open(file_path, "a") as f:
            for i in range(len(x)):
                
                f.write(f"{x[i]},{y[i]},{r[i]},{time[i]}\n")
                
    # a straight line (y=Ax+B) function used for fitting
    def f(self, x, A, B): 
        return A*x + B
            
    def plot_glider_measurements(self, filename):
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
            
        # Create an empty list to store input data
        input_data = []        

        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.extend(line.strip(" []\n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
            
        # Make empty lists to store data
        x_cm = []
        y_cm = []
        r_cm = []
        time = []
        
        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 4):
            
            # Obtain vlaue from input data
            x = float(input_data[i])
            y = float(input_data[i+1])
            r = float(input_data[i+2])
            t = float(input_data[i+3])
            
            # Append to lists
            x_cm.append(x)
            y_cm.append(y)
            r_cm.append(r)
            time.append(t)
            
        
        fig, ax = plt.subplots(1,3, figsize = (16,4.5)) 
        plt.rcParams['font.size'] = 10
        t = np.arange(0,len(r_cm),1)   
        
        # Fit the x-position data and time to a straight line
        popt, pcov = curve_fit(self.f, time, x_cm)
        ax[0].set_title("Glider X-position versus time")             
        ax[0].set_ylabel("Glider X-position")   
        ax[0].set_xlabel("Time")  
        label = f'y = {popt[0]:.3f}t + {popt[1]:.3f}'
        ax[0].plot(x_cm, 'bs', markersize=2, label=label)
        ax[0].legend(loc="upper right")
        
        # Fit the y-position data and time to a straight line
        popt, pcov = curve_fit(self.f, time, y_cm)
        ax[1].set_title("Glider Y-position versus time")             
        ax[1].set_ylabel("Glider Y-position")   
        ax[1].set_xlabel("Time")
        label = f'y = {popt[0]:.3f}t + {popt[1]:.3f}'
        ax[1].plot(y_cm, 'ro', markersize=2, label=label)
        ax[1].legend(loc="upper right")
        
        # Fit the r-position data and time to a straight line
        popt, pcov = curve_fit(self.f, time, r_cm)
        ax[2].set_title("Glider R-position versus time")             
        ax[2].set_ylabel("Glider R-position")   
        ax[2].set_xlabel("Time") 
        label = f'y = {popt[0]:.3f}t + {popt[1]:.3f}' 
        ax[2].plot(r_cm, 'yo', markersize=2, label=label)
        ax[2].legend(loc="upper right")
        
        # Fix any overlapping labels, titles or tick marks
        plt.tight_layout()
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plot successfully saved to: {save_path}")
        
        # Show final plots
        plt.show()
            
    def plot_equilibrium_distribution(self, filename):
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
            
        # Create empty plots
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 10))

        # Create an empty list to store input data
        input_data = []        

        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.extend(line.strip(" \n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
            
        # Make empty lists to store data
        equilibriation_time = []
        no_active_sites = []
        
        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 2):
            
            # Obtain vlaue from input data
            time = float(input_data[i])
            active_sites = float(input_data[i+1])
            
            # Append to lists
            equilibriation_time.append(time)
            no_active_sites.append(active_sites)
            
        # Calculate weighting for histogram
        weights = np.ones_like(equilibriation_time) / len(equilibriation_time) * 100
        
        # Plot a histogram of the equilibriation time
        ax1.hist(equilibriation_time, weights = weights, bins = 30,
                color = "skyblue", edgecolor = "black")
        ax1.set_ylabel("Probability (%)")
        ax1.set_xlabel("Time to equilibriate")
        ax1.set_title("Distribution of the time to equilibriation")
        ax1.grid(True)
        
        # Fix any overlapping labels, titles or tick marks
        plt.tight_layout()
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plot successfully saved to: {save_path}")
        
        # Show final plots
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellular Automata: Game of Life")

    # User input parameters
    parser.add_argument("--n", type = int, default = 50, help = "Lattice size (n x n)")
    parser.add_argument("--steps", type = int, default = 1000, help = "Number of simulation steps")
    parser.add_argument("--init", type = str, default = "random", 
                        choices = ["random", "randomglider", "orderedglider", "blinker"],
                        help = "Initial state for Game of Life")
    parser.add_argument("--type", type = str, default = "ani", 
                        choices = ["ani", "mea"],
                        help = "Simulation or measurements")
    parser.add_argument("--p_alive", type = float, default = 0.5, help = "Probability of a site being alive")

    args = parser.parse_args()
        
    # Game of Life uses n, init, and steps
    sim = Simulation(n = args.n, init_cond = args.init, steps = args.steps)
    
    if args.init == "random" and args.type == "mea":
        
        filename = "game_of_life_equilibrium_distribution_500_systems_1.txt" # Change the name
        sim.equilibrium_measurements(filename, steps = args.steps)
        sim.plot_equilibrium_distribution(filename)
        
    elif (args.init == "orderedglider" or args.init == "randomglider") and args.type == "mea":
        
        filename = "game_of_life_glider_measurements_ordered_glider_100_steps_1.txt"
        #sim.glider_measurements(filename = filename, steps = 100)
        sim.plot_glider_measurements(filename)
        
    else:
    
        sim.animate(steps = args.steps)
    