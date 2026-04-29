#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:50:48 2026

@author: gracetait
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import argparse
import matplotlib.patches as mpatches
from numba import njit

@njit
def sirs_step_numba(lattice, i, j, n, p):
    """
    Determine the next state of a specific cell (i, j) using Monte Carlo rules.
    
    Parameters
    ----------
    lattice : numpy.ndarray
        A 2D array of shape (n, n).
    i : int
        Lattice coordinate of the cell.
    j : int
        Lattice coordinate of the cell
    n : int
        Dimension of the square lattice (n x n)
    p : float
        Probability of infection.

    Returns
    -------
    int
        The new state of the cell.
    """
    
    # Obtain the site
    cell = lattice[i, j]
    
    # First, check if the cell is inactive
    if cell == 0: 
        
        # The cell remains inactive 
        return i, j, 0
        
    # Check if the cell is active ...
    if cell == 1:
        
        # The cell well become inactive with probability p
        if np.random.random() < (1-p):
            
            return i, j, 0
        
        # Else, the cell will infect one of its nearest neighbours randomly
        else:
            
            # Randomly choose neighbour
            # If neighbour is already active, it remains active
            # If neighbour is inactive, it becomes active
            choice = np.random.randint(0, 4)
            
            if choice == 0:
                return (i - 1) % n, j, 1
            
            if choice == 1:
                return (i + 1) % n, j, 1
            
            if choice == 2:
                return i, (j - 1) % n, 1
            
            if choice == 3:
                return i, (j + 1) % n, 1

@njit
def sirs_sweep_numba(lattice, n, N, p):
    """
    Performs one full Monte Carlo Sweep (N random updates) across the lattice.

    Parameters
    ----------
    lattice : numpy.ndarray
        A 2D array of shape (n, n).
    n : int
        Dimension of the square lattice (n x n).
    n : int
        One sweep of the lattice (n x n).
    p : float
        Probability of infection.

    Returns
    -------
    numpy.ndarray
        The updated lattice 
    """
    
    # Iterate N times through the lattice
    for _ in range(N):
        
        # Choose a random site (i, j) in the lattice of size (n x n)
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        
        # Check for updates
        x, y, state = sirs_step_numba(lattice, i, j, n, p)
        
        # Update lattice
        lattice[x, y] = state
        
    return lattice

class SIRS(object):
    """
    A class to represent a Susceptible-Infected-Recovered-Susceptible (SIRS)
    model on a 2D lattice.
    """
    
    def __init__(self, n, p, init_cond):
        """
        Initialise the SIRS lattice parameters

        Parameters
        ----------
        n : int
            Dimension of the square lattice (n x n).
        p : float
            Probability of infection.

        Returns
        -------
        None.

        """
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of sites in the lattice
        self.lattice = None
        self.p = p # Probability of infection.
        self.init_cond = init_cond
        
    def initialise(self):
        """
        Randomly initialise the lattice states based on normalised transition probabilities.
        States are mapped as: 0 (Inactive), 1 (Active).

        Returns
        -------
        None.

        """
        
        if self.init_cond == 'random':
        
            # Create a two-dimensional square lattice according to probabilities
            # Where 0 is inactive, 1 is active
            # The probability of 0 or 1 is equal
            self.lattice = np.random.choice([0, 1], size = (self.n, self.n), p = [0.5, 0.5]).astype(np.int32)
            
        if self.init_cond == 'survival':
            
            # Create a two-dimensional square lattice with 1 random active cell 
            self.lattice = np.zeros((self.n, self.n)).astype(np.int32)
            
            # Choose a random site
            i = np.random.randint(self.n)
            j = np.random.randint(self.n)
            
            # Put one active site in the centre 
            self.lattice[i, j] = 1
            
    def update_lattice(self):
        """
        Perform one full Monte Carlo Sweep using Numba.
        """
        
        # Call Numba function
        self.lattice = sirs_sweep_numba(self.lattice, self.n, self.N, self.p)
        
    def count_infected(self):
        """
        Count the total number of active cells currently in the lattice.

        Returns
        -------
        int
            Number of cells with state 1.

        """
    
        # Count and return the total number of active cells in the lattice
        return np.count_nonzero(self.lattice == 1)
            

class Simulation(object):
    """
    A class to handle the execution, measurement, and visualisation 
    of the SIRS simulation.
    """
    
    def __init__(self, n, steps, p, init_cond):
        """
        Initialise the simulation environment.

        Parameters
        ----------
        n : int
            Lattice dimension.
        steps : int
            Number of measurement steps or animation frames.
        p : float
            Probability of infection.

        Returns
        -------
        None.

        """
        
        # Defining parameters for the lattice
        self.n = n # Size of the two-dimensional square lattice
        self.N = n * n # Total number of sites in the square lattice
        self.steps = steps # Measurement steps or animation frames 
        self.p = p # Probability of infection.
        self.init_cond = init_cond
    
    def animate(self):
        """
        Run and display an animation of the SIRS model spreading.

        Parameters
        ----------
        steps : int
            Total number of animation frames.

        Returns
        -------
        None.

        """
        
        # Initialise the lattice using the SIRS class
        sirs = SIRS(self.n, self.p, self.init_cond)
        sirs.initialise()
        
        # Define the figure and axes for the animation
        fig, ax = plt.subplots()
        
        # Define custom cmap 
        # Use 4 colors: Grey (Vaccinated), Red (Infected), Black (Susceptible), Green (Recovered)
        sirs_cmap = ListedColormap(["#95a5a6", "#e74c3c", "#2c3e50", "#27ae60"])
        
        
        # Define custom cmap and colors
        # Values: 0 (Inactive), 1 (Active)
        colors = ["#2c3e50", "#27ae60"]
        labels = ["Inactive", "Active"]
        sirs_cmap = ListedColormap(colors)
        
        # Create legend handles manually
        # Use mpatches.Patch to create colored squares for the legend
        legend_handles = [mpatches.Patch(color=colors[i], label=labels[i]) 
                          for i in range(len(colors))]
        ax.legend(handles=legend_handles, loc='upper left', 
              bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.subplots_adjust(right=0.75)
        
        # Initialise the image object
        # vmin/vmax ensure 0 (inactive) is black and 1 (active) is green consistently
        im = ax.imshow(sirs.lattice, cmap = sirs_cmap, vmin = 0, vmax = 1)
        
        # Run the animation for the total number of steps
        for s in range(self.steps):
            
            # Update lattice
            sirs.update_lattice()
            
            # Update the animation 
            im.set_data(sirs.lattice)
            ax.set_title(f"Step: {s}")
            
            # Keep the image up while the script is running
            plt.pause(0.001)
            
        # Keep the final image open when the loop finishes
        plt.show()
        
    def calculate_average_active(self, tot_active_list):
        """
        Calculate the mean fraction of active sites from a list of measurements.

        Parameters
        ----------
        tot_active_list : list of int
            List containing counts of active sites.

        Returns
        -------
        float
            The average active fraction <I>/N.

        """
        
        # Calculate and return the mean fraction of active sites
        return np.mean(tot_active_list) / self.N
    
    def active_sites_measurements(self, filename):
        """
        Perform measurements of number of active sites over time.

        Parameters
        ----------
        filename : str
            Output test file name.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders dont exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
        
        # Initialise the lattice using the SIRS class
        sirs = SIRS(self.n, self.p, self.init_cond)
        sirs.initialise()
        
        # Make an empty list to hold results
        fraction_active = []
        
        # Iterate through steps
        for s in range(self.steps):
            
            # Update lattice
            sirs_sweep_numba(sirs.lattice, sirs.n, sirs.N, sirs.p)
            
            # Count active cells
            active_count = sirs.count_infected()
            
            # Optimisation: If the virus dies out, it stays dead
            if active_count == 0:
                
                # Fill the rest of the list with 0 and break
                remaining = self.steps - len(fraction_active)
                fraction_active.extend([0] * remaining)
                break
            
            # Count fraction of active cells
            fraction = self.calculate_average_active(active_count)
            fraction_active.append(fraction)
            
        # Write the values into the specified file
        with open(file_path, "w") as f:
            for s in range(self.steps):
                f.writelines(f"{fraction_active[s]},{s}\n")
                
    def plot_active_sites(self, filename):
        """
        Generate and save a plot of the fraction of active agents <I>/N  versus time.

        Parameters
        ----------
        filename : str
            The data file to read from.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        
        # Create empty plots
        fig, ax = plt.subplots(figsize=(8, 10))
        
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
        
        # Create empty lists to hold data 
        frac = []
        time = []
        
        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 2):
            
            # Obtain value from input data
            f = float(input_data[i])
            t = float(input_data[i+1])

            # Append to empty lists
            frac.append(f)
            time.append(t)
        
        # Plot graph
        ax.plot(time, frac, 'o-', 
                color='red', markerfacecolor='black', markeredgecolor='black',
                markersize=2, linewidth=1.5, label=r'$\langle I \rangle / N$')
        
        # Labels and formatting
        ax.set_xlabel(r"Timesteps", fontsize=14)
        ax.set_ylabel(r"Fraction of Active Cells $\langle I \rangle / N$", fontsize=14)
        ax.set_title(rf"Fraction of Active Cells Versus Time with ($p={self.p}$)", fontsize=16)
        
        # Add a horizontal line at 0 to show when the infection is gone
        ax.axhline(0, color='black', lw=0.8, linestyle='--')
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
        ax.legend()

        plt.tight_layout()
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plot successfully saved to: {save_path}")
        
        # Show final plots
        plt.show()
        
    def total_active_sites_measurements(self, filename):
        """
        Measure the average active fraction as a function of the probability 
        of infection.

        Parameters
        ----------
        filename : str
            Output text file name.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders dont exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
         
        # Define array of probability of infection
        p_array = np.round(np.arange(0.55, 0.7, 0.005), 3)
        
        # Create empty list to hold results 
        results = []
        
        # Iterate through fraction immunity array
        for p in p_array:
            print(f"\rSimulating p = {p}...", end="", flush=True)
               
            # Initialise the lattice using the SIRS class
            sirs = SIRS(self.n, p, self.init_cond)
            sirs.initialise()
            
            # Make an empty list to hold data
            active_list = []
            
            # Equilibrate for 100 sweeps
            for _ in range(100):
                sirs.update_lattice()
            
            # Measure for self.steps sweeps
            for _ in range(self.steps):
                sirs.update_lattice()
                active_count = np.count_nonzero(sirs.lattice == 1)
                active_list.append(active_count)
                
                if active_count == 0:
                    remaining = self.steps - len(active_list)
                    active_list.extend([0] * remaining)
                    break
                
            # After completing all the measurements for the probabilities
            # Calcaulte the average and the variance of the fraction of active sites
            mean_active = self.calculate_average_active(active_list)
            
            # Write results to results list
            results.append(f"{p},{mean_active}\n")
            
        # Write the values into the specified file
        with open(file_path, "w") as f:
            f.writelines(results)
 
    def plot_total_active_sites(self, filename):
        """
        Generate and save a plot of <I>/N versus the fraction of active agents.

        Parameters
        ----------
        filename : str
            The data file to read from.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        
        # Create empty plots
        fig, ax = plt.subplots(figsize=(8, 10))
        
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
        
        # Create empty lists to hold data 
        p_array = []
        active_sites = []
        
        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 2):
            
            # Obtain value from input data
            p = float(input_data[i])
            a = float(input_data[i+1])

            # Append to empty lists
            p_array.append(p)
            active_sites.append(a)
        
        # Plot variance with error bars using the bootstrap results
        ax.plot(p_array, active_sites, 'o-', 
                color='red', markerfacecolor='black', markeredgecolor='black',
                markersize=4, linewidth=1.5, label=r'$\langle I \rangle / N$')
        
        # Labels and formatting
        ax.set_xlabel(r"Probability of Infection $p$", fontsize=14)
        ax.set_ylabel(r"Average Active Sites Fraction $\langle A \rangle / N$", fontsize=14)
        ax.set_title(r"Average Active Sites Fraction Versus Probability of Infection $p$", fontsize=16)
        
        # Add a horizontal line at 0 to show when the infection is gone
        ax.axhline(0, color='black', lw=0.8, linestyle='--')
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
        ax.legend()

        plt.tight_layout()
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plot successfully saved to: {save_path}")
        
        # Show final plots
        plt.show()
        
    def calculate_variance_active(self, tot_active_list):
        """
        Calculate the normalised variance of the active population.

        Parameters
        ----------
        tot_active_list : list of int
            Counts of active sites.

        Returns
        -------
        float
            The normalized variance: (<I^2> - <I>^2)/N.

        """
        
        # Convert to numpy array
        tot_active_list = np.array(tot_active_list)
        
        # Calculate and return the variance
        mean_active_squared = np.mean(tot_active_list**2)
        mean_active = np.mean(tot_active_list)
        return (mean_active_squared - mean_active**2) / self.N
    
    def bootstrap_method(self, data):
        """
        Estimate the standard error of the variance using the Bootstrap resampling method.

        Parameters
        ----------
        data : list or np.ndarray
            The population data to resample.

        Returns
        -------
        float
            Standard deviation of the resampled variances.

        """
        
        # Convert to numpy array
        data = np.array(data)
        
        # Find the length of the data
        n = len(data)
        
        # Create an empty list to store the resampled values
        resampled_values = []
        
        # Resampling 1000 times is sufficient
        for j in range(1000):
            
             # Randomnly resample from the n measurements
             ind = np.random.randint(0, n, size = n)
             resample = data[ind]
             
             # Calculate specific heat accordingly
             mean_E_sq = np.mean(resample**2)
             mean_E = np.mean(resample)
             value = (mean_E_sq - mean_E**2) / self.N
             
             # Append to the list
             resampled_values.append(value)
        
        # Calculate and return the error
        # Which is the standard deviation of the resampled values
        return np.std(np.array(resampled_values))
            
    def variance_measurements(self, filename):
        """
        Measure the variance of the average active fraction as a function of 
        the probability of infection.

        Parameters
        ----------
        filename : str
            Output text file name.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders dont exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
         
        # Define array of probability of infection
        p_array = np.round(np.arange(0.55, 0.7, 0.005), 3)
        
        # Create empty list to hold results 
        results = []
        
        # Iterate through fraction immunity array
        for p in p_array:
            print(f"\rSimulating p = {p}...", end="", flush=True)
               
            # Initialise the lattice using the SIRS class
            sirs = SIRS(self.n, p, self.init_cond)
            sirs.initialise()
            
            # Make an empty list to hold data
            active_list = []
            
            # Equilibrate for 100 sweeps
            for _ in range(100):
                sirs.update_lattice()
            
            # Measure for self.steps sweeps
            for _ in range(self.steps):
                sirs.update_lattice()
                active_count = np.count_nonzero(sirs.lattice == 1)
                active_list.append(active_count)
                
                if active_count == 0:
                    remaining = self.steps - len(active_list)
                    active_list.extend([0] * remaining)
                    break
                
            # After completing all the measurements for the probabilities
            # Calcaulte the average and the variance of the fraction of active sites
            variance_active = self.calculate_variance_active(active_list)
            variance_active_err = self.bootstrap_method(active_list)
            
            # Write results to results list
            results.append(f"{p},{variance_active},{variance_active_err}\n")
            
        # Write the values into the specified file
        with open(file_path, "w") as f:
            f.writelines(results)
            
    def plot_variance(self, filename):
        """
        Generate and save a plot of normalised variance with error bars.

        Parameters
        ----------
        filename : str
            The data file to read from.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        
        # Create empty plots
        fig, ax = plt.subplots(figsize=(8, 10))
        
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
        
        # Create emptry lists to hold data 
        p_array = []
        variance_active_array = []
        variance_active_err_array = []
        
        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 3):
            
            # Obtain vlaue from input data
            p = float(input_data[i])
            var = float(input_data[i+1])
            var_err = float(input_data[i+2])

            # Append to empty lists
            p_array.append(p)
            variance_active_array.append(var)
            variance_active_err_array.append(var_err)
        
        # Plot variance with error bars using the bootstrap results
        ax.errorbar(p_array, variance_active_array, yerr=variance_active_err_array, 
                    fmt='o-', color='red', ecolor='black', markerfacecolor = 'black', markeredgecolor = 'black',
                    capsize=3, elinewidth=1, markeredgewidth=1, markersize = 4, label = 'Infected Variance')
        
        # Labels and formatting to show the phase transition
        ax.set_xlabel(r"Infection Probability $p$", fontsize=14)
        ax.set_ylabel(r"Normalised Variance $(\langle A^2 \rangle - \langle A \rangle^2)/N$", fontsize=14)
        ax.set_title(r"Normalised Variance vs Infection Probability", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
        ax.legend()
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plot successfully saved to: {save_path}")
        
        # Show final plots
        plt.show()

    def survival_measurements(self, filename, sims):
        """
        Perform measurements of number of active sites over time.

        Parameters
        ----------
        filename : str
            Output test file name.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders dont exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
            
        # Create empty list to hold results
        total_fraction_active_array = []
        
        # Iterate through sims simulations
        for _ in range(sims):
            
            # Initialise the lattice using the SIRS class
            sirs = SIRS(self.n, self.p, self.init_cond)
            sirs.initialise()
            
            # Make an empty list to hold results
            fraction_active = []
            
            # Iterate through number of steps
            for s in range(self.steps):
                
                # Update lattice
                sirs.update_lattice()
                
                # Count active cells
                active_count = sirs.count_infected()
                
                # Optimisation: If the virus dies out, it stays dead
                if active_count == 0:
                    
                    # Fill the rest of the list with 0 and break
                    remaining = self.steps - len(fraction_active)
                    fraction_active.extend([0] * remaining)
                    break
                
                # Count fraction of active cells
                fraction = self.calculate_average_active(active_count)
                fraction_active.append(fraction)
                
            # Append to array
            total_fraction_active_array.append(fraction_active)
        
        # Empty list to hold data
        p_survival_list = []
        
        # Compute the survival probability 
        for s in range(self.steps):
            
            # Start survival count at zero
            survived_count = 0
            
            # Count number of simulations with survived sites at this timestep
            for act in total_fraction_active_array:
                
                if act[s] > 0: 
                    
                    survived_count += 1
                    
            # Compute probability
            p_survival = survived_count / sims
            
            # Append to list
            p_survival_list.append(p_survival)
            
        # Write the values into the specified file
        with open(file_path, "w") as f:
            for s in range(self.steps):
                t = s + 1 # Start time from 1 to avoid log(0) error
                f.writelines(f"{p_survival_list[s]},{t}\n")
                
    def plot_survival(self, filename):
        """
        Generate and save a plot of probability of survival versus time.

        Parameters
        ----------
        filename : str
            The data file to read from.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        
        # Create empty plots
        fig, ax = plt.subplots(figsize=(8, 10))
        
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
        
        # Create empty lists to hold data 
        prob = []
        time = []
        
        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 2):
            
            # Obtain value from input data
            p = float(input_data[i])
            t = float(input_data[i+1])

            # Append to empty lists
            prob.append(p)
            time.append(t)
        
        # Plot graph
        ax.plot(np.log(time), np.log(prob), 'o-', 
                color='red', markerfacecolor='black', markeredgecolor='black',
                markersize=2, linewidth=1.5, label=r'$\langle I \rangle / N$')
        
        # Labels and formatting
        ax.set_xlabel(r"log(Time)", fontsize=14)
        ax.set_ylabel(r"log(Probability of Survival)", fontsize=14)
        ax.set_title(rf"log(Probability of Survival) Versus log(Time) for p={args.p}", fontsize=16)
        
        # Add a horizontal line at 0 to show when the infection is gone
        ax.axhline(0, color='black', lw=0.8, linestyle='--')
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
        ax.legend()

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
    parser = argparse.ArgumentParser(description="Cellular Automata: SIRS")

    # User input parameters
    parser.add_argument("--n", type=int, default=50, help="Lattice size (n x n)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of measurement steps or animation frames.")
    parser.add_argument("--sims", type=int, default=1000, help="Number of simulations for the survival probability measurements.")
    parser.add_argument("--p", type=float, default=0.3, help="Infection probability")
    parser.add_argument("--mode", type = str, default = "ani", 
                        choices = ["ani", "mea"],
                        help = "Animation or measurements")
    parser.add_argument("--measure", type = str, default = "activevstime", 
                         choices = ["activevstime", "activevsp", "variance", "survival"])

    args = parser.parse_args()
        
    if args.mode == "ani":
        
        sim = Simulation(n=args.n, steps=args.steps, p=args.p, init_cond="random")
        sim.animate()
        
    if args.mode == "mea" and args.measure == "activevstime":
        
        filename = f"fraction_of_active_cells_{args.steps}steps_{args.p}p.txt"
        sim = Simulation(n=args.n, steps=args.steps, p=args.p, init_cond="random")
        sim.active_sites_measurements(filename)
        sim.plot_active_sites(filename)
        
    if args.mode == "mea" and args.measure == "activevsp":
        
        filename = f"fraction_of_average_active_cells_vs_p_{args.steps}steps.txt"
        sim = Simulation(n=args.n, steps=args.steps, p=args.p, init_cond="random")
        sim.total_active_sites_measurements(filename)
        sim.plot_total_active_sites(filename)
        
    if args.mode == "mea" and args.measure == "variance":
        
        filename = f"variance_of_average_active_cells_vs_p_{args.steps}steps.txt"
        sim = Simulation(n=args.n, steps=args.steps, p=args.p, init_cond="random")
        sim.variance_measurements(filename)
        sim.plot_variance(filename)
        
    if args.mode == "mea" and args.measure == "survival":
        
        filename = f"survival_probability_{args.steps}simulations_{args.p}p.txt"
        sim = Simulation(n=args.n, steps=args.steps, p=args.p, init_cond="survival")
        sim.survival_measurements(filename, sims=args.sims)
        sim.plot_survival(filename)
        
        
        
        