#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 11:54:13 2026

@author: gracetait
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import argparse

class SIRS(object):
    
    def __init__(self, n, p_S, p_I, p_R):
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of sites in the lattice
        self.lattice = None
        self.p_S = p_S # Probability of susceptible to infected
        self.p_I = p_I # Probability of infected to recovered
        self.p_R = p_R # Probability of recovered to susceptible
        
    def initialise(self):
        
        # Create a two-dimensional square lattice
        # Where -1 is infected, 0 is susceptible, 1 is alive
        self.lattice = np.random.choice([-1,0, 1], size = (self.n, self.n)).astype(int)
        
    def infected_or_susceptible_or_recovered(self, i, j, p_S, p_I, p_R):
        
        # Obtain the site 
        cell = self.lattice[i, j]
        
        # Collect all of the nearest neighbours
        nearest_neighbours = [self.lattice[(i - 1) % self.n, j],
                              self.lattice[(i + 1) % self.n, j],
                              self.lattice[i, (j - 1) % self.n],
                              self.lattice[i, (j + 1) % self.n]
            ]
        
        # Start with all counts at zero
        infected = 0
        susceptible = 0
        recovered = 0
        
        # Count all neighbours to see if they are infected, susceptible or alive
        for nn in nearest_neighbours:
            
            if nn == -1:
                infected += 1
                
            elif nn == 0:
                susceptible += 1
                
            else:
                recovered += 1
        
        # If the cell is susceptible ...
        if cell == 0:
            
            # If there is at least infected nearest neighbours
            if infected >= 1: 
                
                # The cell will be infected with probability p_S
                return np.random.choice([0, -1], p = [1 - p_S, p_S])
            
        # If the cell is infected ...
        elif cell == -1:
            
            # The cell will be recovered with probability p_I
            return np.random.choice([-1, 1], p = [1 - p_I, p_I])
            
        # If the cell is recovered ...
        else:
            
            # The cell will be susceptible with probability p_S
            return np.random.choice([1, 0], p = [1 - p_R, p_R])
        
    def update_lattice(self):
        
        # Keep old lattice to check for updates
        lattice_new = np.zeros((self.n, self.n))
        
        # Iterate through the whole lattice
        # Update the new lattice at each iteration
        for i in range(self.n):
            
            for j in range(self.n):
            
                lattice_new[i, j] = self.infected_or_susceptible_or_recovered(i, j, self.p_S, self.p_I, self.p_R)
                
        # Once done, update the main lattice simultaneously
        self.lattice = lattice_new
                
        return self.lattice
            

class Simulation(object):
    
    def __init__(self, n, steps, p_S, p_I, p_R):
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of sites in the square lattice
        self.steps = steps # Choice of number of steps for the animation
        self.p_S = p_S # Probability of susceptible to infected
        self.p_I = p_I # Probability of infected to recovered
        self.p_R = p_R # Probability of recovered to susceptible
    
    def animate(self, steps):
        
        # Initialise the lattice using the SIRS class
        sirs = SIRS(self.n, self.p_S, self.p_I, self.p_R)
        sirs.initialise()
        
        # Define the figure and axes for the animation
        fig, ax = plt.subplots()
        
        # Define custom cmap
        sirs_cmap = ListedColormap(["#e74c3c", "#2c3e50", "#27ae60"])
        
        # Initialise the image object
        # vmin/vmax ensure -1 is red, 0 is black, and 1 is green consistently
        im = ax.imshow(sirs.lattice, cmap = sirs_cmap,
                       vmin = -1, vmax = 1)
        
        # Run the animation for the total number of steps
        for s in range(steps):
            
            # Update lattice
            sirs.update_lattice()
            
            # Update the animation 
            im.set_data(sirs.lattice)
            ax.set_title(f"Step: {s}")
            
            # Keep the image up while the script is running
            plt.pause(0.001)
            
        # Keep the final image open when the loop finishes
        plt.show()
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellular Automata: SIRS")

    # User input parameters
    parser.add_argument("--n", type=int, default=50, help="Lattice size (n x n)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--p_S", type=float, default=0.5, help="S -> I infection probability")
    parser.add_argument("--p_R", type=float, default=0.2, help="I -> R recovery probability")
    parser.add_argument("--p_I", type=float, default=0.01, help="R -> S immunity loss probability")

    args = parser.parse_args()

    sim = Simulation(n=args.n, 
                     steps=args.steps,
                     p_S=args.p_S, 
                     p_R=args.p_R, 
                     p_I=args.p_I)
    
    sim.animate(steps=args.steps)