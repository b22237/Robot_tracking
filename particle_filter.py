import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# =============================================================================
# --- Environment Setup ---
# =============================================================================

# Define the fixed, known landmarks (M x 2) array
landmarks = np.array([
    [10, 10],
    [10, 20],
    [20, 10],
    [20, 20]
])

# =============================================================================
# CLASS 1: THE GROUND TRUTH ROBOT
# =============================================================================
class Robot:
    """
    This class simulates the *ground truth* robot.
    It moves and senses with *actual* (true) noise.
    The filter will *never* see its 'state'.
    """
    def __init__(self, x=0.0, y=0.0, theta=0.0, 
                 d_noise_std=0.1,  # True std dev for distance (e.g., 0.1m)
                 th_noise_std=0.05, # True std dev for turning (e.g., 0.05 rad)
                 r_noise_std=0.5):  # True std dev for range sensor (e.g., 0.5m)
        
        self.state = np.array([x, y, theta]) # [x, y, orientation]
        self.d_noise_std = d_noise_std
        self.th_noise_std = th_noise_std
        self.r_noise_std = r_noise_std

    def move(self, command_d=1.0, command_th=np.deg2rad(10.0)):
        """ Simulates the robot's *actual* movement. """
        d_noisy = command_d + np.random.normal(0, self.d_noise_std)
        th_noisy = command_th + np.random.normal(0, self.th_noise_std)
        
        self.state[0] += d_noisy * np.cos(self.state[2] + th_noisy)
        self.state[1] += d_noisy * np.sin(self.state[2] + th_noisy)
        self.state[2] += th_noisy
        self.state[2] = self.normalize_angle(self.state[2])
        return self.state

    def sense(self, landmarks):
        """ Simulates the robot's *actual* noisy sensor readings. """
        measurements = []
        for (lx, ly) in landmarks:
            true_dist = np.sqrt((self.state[0] - lx)**2 + (self.state[1] - ly)**2)
            noisy_dist = true_dist + np.random.normal(0, self.r_noise_std)
            measurements.append(noisy_dist)
        return np.array(measurements)

    def normalize_angle(self, angle):
        """ Wrap angle to [-pi, pi] """
        return (angle + np.pi) % (2 * np.pi) - np.pi

# =============================================================================
# CLASS 2: THE PARTICLE FILTER (ESTIMATOR)
# =============================================================================
# Paste this replacing your ParticleFilter class (or merge changes)
import numpy as np
from scipy.stats import norm

class ParticleFilter:
    def __init__(self, num_particles,
                 d_noise_filter_std,
                 th_noise_filter_std,
                 r_noise_filter_std,
                 resample_jitter=1e-3):  # small jitter after resample to avoid collapse
        self.N = int(num_particles)
        self.d_std = float(d_noise_filter_std)
        self.th_std = float(th_noise_filter_std)
        self.r_std = float(r_noise_filter_std)
        self.jitter = float(resample_jitter)

        self.particles = np.empty((self.N, 3))
        self.weights = np.full(self.N, 1.0 / self.N)

    def create_uniform_particles(self, x_range, y_range, th_range):
        self.particles[:, 0] = np.random.uniform(x_range[0], x_range[1], self.N)
        self.particles[:, 1] = np.random.uniform(y_range[0], y_range[1], self.N)
        self.particles[:, 2] = np.random.uniform(th_range[0], th_range[1], self.N)
        self.particles[:, 2] = self.normalize_angle_vectorized(self.particles[:, 2])

    def propagate(self, command_d=1.0, command_th=np.deg2rad(10.0)):
        # Predict step: apply noisy command to each particle (turn-then-move model)
        d_noisy = command_d + np.random.normal(0.0, self.d_std, self.N)
        th_noisy = command_th + np.random.normal(0.0, self.th_std, self.N)

        self.particles[:, 0] += d_noisy * np.cos(self.particles[:, 2] + th_noisy)
        self.particles[:, 1] += d_noisy * np.sin(self.particles[:, 2] + th_noisy)
        self.particles[:, 2] += th_noisy
        self.particles[:, 2] = self.normalize_angle_vectorized(self.particles[:, 2])

    def update(self, measurements, landmarks):
        # measurements: array len M
        # landmarks: shape (M,2)
        M = len(landmarks)
        log_w = np.zeros(self.N)
        for i, (lx, ly) in enumerate(landmarks):
            # predicted distances for each particle to landmark i
            dx = self.particles[:, 0] - lx
            dy = self.particles[:, 1] - ly
            dist = np.hypot(dx, dy)
            z = measurements[i]
            # log-likelihood per particle (broadcasting)
            log_w += norm.logpdf(z, loc=dist, scale=self.r_std)

        # stable exponentiation
        max_log = np.max(log_w)
        log_w -= max_log
        w = np.exp(log_w)
        s = np.sum(w)
        if s <= 1e-12:
            # prevent collapse: reset to uniform and optionally jitter particles
            print("Warning: tiny total weight, resetting to uniform weights.")
            self.weights[:] = 1.0 / self.N
        else:
            self.weights = w / s

    def effective_sample_size(self):
        return 1.0 / np.sum(self.weights**2)

    def resample(self, threshold=None):
        """
        Systematic (low-variance) resampling.
        If threshold provided, resample only if ESS < threshold; otherwise always resample.
        Returns True if resampled.
        """
        ess = self.effective_sample_size()
        if (threshold is not None) and (ess >= threshold):
            return False  # no resample

        cumulative = np.cumsum(self.weights)
        # ensure last element exactly 1.0 (avoid numerical issues)
        cumulative[-1] = 1.0
        start = np.random.uniform(0.0, 1.0 / self.N)
        positions = start + (np.arange(self.N) / self.N)
        indices = np.searchsorted(cumulative, positions, side='right')
        # copy particles
        self.particles = self.particles[indices].copy()
        # reset weights uniform
        self.weights.fill(1.0 / self.N)
        # optional small jitter to avoid exact duplicates
        if self.jitter > 0:
            self.particles[:, 0:2] += np.random.normal(scale=self.jitter, size=(self.N, 2))
            self.particles[:, 2] = self.normalize_angle_vectorized(self.particles[:, 2])
        return True

    def get_estimate(self):
        mean_x = np.sum(self.particles[:, 0] * self.weights)
        mean_y = np.sum(self.particles[:, 1] * self.weights)
        s = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        c = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        mean_theta = np.arctan2(s, c)
        return np.array([mean_x, mean_y, mean_theta])

    def get_variance(self):
        mean = self.get_estimate()
        var_x = np.sum(self.weights * (self.particles[:, 0] - mean[0])**2)
        var_y = np.sum(self.weights * (self.particles[:, 1] - mean[1])**2)
        return var_x + var_y

    def normalize_angle_vectorized(self, angles):
        return (angles + np.pi) % (2.0 * np.pi) - np.pi
class Summary:
    def __init__(self):
        self.errors = []
        self.variances = []
        self.ess_values = []
        self.resample_count = 0
        self.diverged = False

    def update(self, true_state, estimate, variance, ess):
        dx = estimate[0] - true_state[0]
        dy = estimate[1] - true_state[1]
        err = np.sqrt(dx*dx + dy*dy)

        self.errors.append(err)
        self.variances.append(variance)
        self.ess_values.append(ess)

        if err > 5:  # divergence threshold
            self.diverged = True

    def final_summary(self, experiment_name):
        return f"""
==== {experiment_name} ====
Initial Error:     {self.errors[0]:.3f}
Final Error:       {self.errors[-1]:.3f}
Mean Error:        {np.mean(self.errors):.3f}

Min Variance:      {np.min(self.variances):.4f}
Max Variance:      {np.max(self.variances):.4f}
Final Variance:    {self.variances[-1]:.4f}

Avg ESS:           {np.mean(self.ess_values):.2f}
Resample Count:    {self.resample_count}

Diverged?:         {"YES ❌" if self.diverged else "NO ✅"}

-------------------------------
"""

# =============================================================================
# --- MAIN SIMULATION AND PLOTTING ---
# =============================================================================
def run_simulation(config):
    """
    Main function to run the simulation and experiments.
    """
    
    # --- 1. Initialization ---
    summary = Summary()

    # Create the Ground Truth Robot
    robot = Robot(
        x=0.0, y=0.0, theta=np.deg2rad(90), # Start at (0,0) facing 'up'
        d_noise_std=config["ROBOT_D_STD"],
        th_noise_std=config["ROBOT_TH_STD"],
        r_noise_std=config["ROBOT_R_STD"]
    )
    
    # Create the Particle Filter
    pf = ParticleFilter(
        num_particles=config["NUM_PARTICLES"],
        d_noise_filter_std=config["FILTER_D_STD"],
        th_noise_filter_std=config["FILTER_TH_STD"],
        r_noise_filter_std=config["FILTER_R_STD"]
    )
    
    # Initialize particle cloud (e.g., around the start position)
    pf.create_uniform_particles(
        x_range=(-5, 5), y_range=(-5, 5), th_range=(0, 2*np.pi)
    )
    
    # Setup history storage for final plots
    history_true_state = []
    history_estimated_state = []
    history_variance = []
    
    # Setup interactive plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 12))

    # --- 2. Main Simulation Loop ---
    
    for step in range(config["SIM_STEPS"]):
        
        # --- Filter Core Steps ---
        
        # 1. Ground Truth Robot moves
        true_state = robot.move()
        
        # 2. Ground Truth Robot senses (filter only sees this)
        measurements = robot.sense(landmarks)
        
        # 3. Filter: Propagate (Predict)
        pf.propagate()
        
        # 4. Filter: Update (Correct)
        pf.update(measurements, landmarks)
        
        # --- Get Metrics & Store History ---
        estimate = pf.get_estimate()
        variance = pf.get_variance()
        # TEXT LOGGING (optional)
        ess = pf.effective_sample_size()
        summary.update(true_state, estimate, variance, ess)

        # print_step_info(step+1, true_state, estimate, variance, pf)

        history_true_state.append(true_state.copy())
        history_estimated_state.append(estimate.copy())
        history_variance.append(variance)
        
        # --- Live Plotting (as required by assignment) ---
        ax.cla() # Clear the axis
        
        # A. Plot all particles (faint)
        ax.scatter(pf.particles[:, 0], pf.particles[:, 1], 
                   color='black', alpha=0.1, s=10, label='Particles')

        # B. Plot weighted particles (size proportional to weight)
        #    This is the "larger circle" requirement, done for all particles
        #    We visualize weights *before* resampling
        ax.scatter(pf.particles[:, 0], pf.particles[:, 1], 
                   color='blue', alpha=0.5, 
                   s=pf.weights * config["NUM_PARTICLES"] * 50, # Scale size
                   label='Weighted Belief')

        # C. Plot landmarks
        ax.plot(landmarks[:, 0], landmarks[:, 1], 'bo', markersize=10, label='Landmarks')
        
        # D. Plot paths (history)
        path_true = np.array(history_true_state)
        path_est = np.array(history_estimated_state)
        ax.plot(path_true[:, 0], path_true[:, 1], 'r-', label='True Path')
        ax.plot(path_est[:, 0], path_est[:, 1], 'g--', label='Estimated Path')
        
        # E. Plot current positions
        ax.plot(true_state[0], true_state[1], 'ro', markersize=12, label='True Robot')
        ax.plot(estimate[0], estimate[1], 'go', markersize=12, label='Estimate')
        
        ax.set_title(f'Step: {step+1}/{config["SIM_STEPS"]} | Particle Variance: {variance:.4f}')
        ax.legend(loc='upper right')
        ax.set_xlim(-5, 25)
        ax.set_ylim(-5, 25)
        
        plt.pause(0.01)
        
        # --- 5. Filter: Resample ---
        # This is the *last* step, ready for the next loop
        # resample only if ESS < N/2
        # pf.resample(threshold=pf.N / 2.0)
        if pf.resample(threshold=pf.N / 2.0):
            summary.resample_count += 1

    # return summary



    # --- 3. Final Plots ---
    plt.ioff() # Turn off interactive mode
    
    # Plot 1: Final Simulation State (already showing)
    ax.set_title(f'Final State | Particle Variance: {variance:.4f}')
    plt.show()
    
    # Plot 2: Variance over Time
    fig_var, ax_var = plt.subplots()
    ax_var.plot(history_variance)
    ax_var.set_title('Particle Cloud Variance Over Time')
    ax_var.set_xlabel('Simulation Step')
    ax_var.set_ylabel('Sum of X, Y Variance')
    plt.show()
    plt.ioff()
    # --- 3. Final Plots ---
    # plt.ioff()  # Turn off interactive mode

    # 1. Save Final Particle Plot
    fig.savefig(f"{config['NAME']}_final_particles.png", dpi=300)

    # 2. Variance Plot
    fig_var, ax_var = plt.subplots()
    ax_var.plot(history_variance, linewidth=2)
    ax_var.set_title('Particle Cloud Variance Over Time')
    ax_var.set_xlabel('Simulation Step')
    ax_var.set_ylabel('Sum of X, Y Variance')

    fig_var.savefig(f"{config['NAME']}_variance.png", dpi=300)

    plt.close(fig)
    plt.close(fig_var)

    return summary

    # return summary
def run_all_experiments():
    # ====================================================
    # EXPERIMENT 1 — EFFECT OF PARTICLE COUNT (N)
    # ====================================================

    print("\n===== Experiment 1.1: Low Particle Count (N=20) =====\n")
    Config_1_1 = {
        "NUM_PARTICLES": 20,
        "SIM_STEPS": 50,
        "ROBOT_D_STD": 0.1,
        "ROBOT_TH_STD": 0.05,
        "ROBOT_R_STD": 0.5,
        "FILTER_D_STD": 0.1,
        "FILTER_TH_STD": 0.05,
        "FILTER_R_STD": 0.5,
        "NAME": "Experiment_1_1"
    }
    summary = run_simulation(Config_1_1)
    print(summary.final_summary("Experiment 1.1 — N=20"))


    print("\n===== Experiment 1.2: High Particle Count (N=1000) =====\n")
    Config_1_2 = {
        "NUM_PARTICLES": 1000,
        "SIM_STEPS": 50,
        "ROBOT_D_STD": 0.1,
        "ROBOT_TH_STD": 0.05,
        "ROBOT_R_STD": 0.5,
        "FILTER_D_STD": 0.1,
        "FILTER_TH_STD": 0.05,
        "FILTER_R_STD": 0.5,
        "NAME": "Experiment_1_2"
    }
    summary = run_simulation(Config_1_2)
    print(summary.final_summary("Experiment 1.2 — N=1000"))


    # ====================================================
    # EXPERIMENT 2 — EFFECT OF PROCESS NOISE (Q)
    # ====================================================

    print("\n===== Experiment 2.1: Overconfident Filter (Q too small) =====\n")
    Config_2_1 = {
        "NUM_PARTICLES": 100,
        "SIM_STEPS": 50,
        "ROBOT_D_STD": 0.1,
        "ROBOT_TH_STD": 0.05,
        "ROBOT_R_STD": 0.5,
        "FILTER_D_STD": 0.01,
        "FILTER_TH_STD": 0.005,
        "FILTER_R_STD": 0.5,
        "NAME": "Experiment_2_1"
    }
    summary = run_simulation(Config_2_1)
    print(summary.final_summary("Experiment 2.1 — Overconfident Q"))


    print("\n===== Experiment 2.2: Cautious Filter (Q too large) =====\n")
    Config_2_2 = {
        "NUM_PARTICLES": 100,
        "SIM_STEPS": 50,
        "ROBOT_D_STD": 0.1,
        "ROBOT_TH_STD": 0.05,
        "ROBOT_R_STD": 0.5,
        "FILTER_D_STD": 1.0,
        "FILTER_TH_STD": 0.5,
        "FILTER_R_STD": 0.5,
        "NAME": "Experiment_2_2"
    }
    summary = run_simulation(Config_2_2)
    print(summary.final_summary("Experiment 2.2 — Cautious Q"))


    # ====================================================
    # EXPERIMENT 3 — EFFECT OF MEASUREMENT NOISE (R)
    # ====================================================

    print("\n===== Experiment 3.1: Over-Trusting Filter (R too small) =====\n")
    Config_3_1 = {
        "NUM_PARTICLES": 100,
        "SIM_STEPS": 50,
        "ROBOT_D_STD": 0.1,
        "ROBOT_TH_STD": 0.05,
        "ROBOT_R_STD": 0.5,
        "FILTER_D_STD": 0.1,
        "FILTER_TH_STD": 0.05,
        "FILTER_R_STD": 0.05,
        "NAME": "Experiment_3_1"
    }
    summary = run_simulation(Config_3_1)
    print(summary.final_summary("Experiment 3.1 — Over-Trusting R"))


    print("\n===== Experiment 3.2: Skeptical Filter (R too large) =====\n")
    Config_3_2 = {
        "NUM_PARTICLES": 100,
        "SIM_STEPS": 50,
        "ROBOT_D_STD": 0.1,
        "ROBOT_TH_STD": 0.05,
        "ROBOT_R_STD": 0.5,
        "FILTER_D_STD": 0.1,
        "FILTER_TH_STD": 0.05,
        "FILTER_R_STD": 5.0,
        "NAME": "Experiment_3_2"
    }
    summary = run_simulation(Config_3_2)
    print(summary.final_summary("Experiment 3.2 — Skeptical R"))

def print_step_info(step, true_state, estimate, variance, pf):
    """
    Printable text output for checking correctness.
    You can paste this output directly into ChatGPT.
    """
    err_x = estimate[0] - true_state[0]
    err_y = estimate[1] - true_state[1]
    err = np.sqrt(err_x**2 + err_y**2)

    print(f"--- Step {step} ---")
    print(f"True State:      x={true_state[0]:.3f}, y={true_state[1]:.3f}, th={true_state[2]:.3f}")
    print(f"Estimate:        x={estimate[0]:.3f}, y={estimate[1]:.3f}, th={estimate[2]:.3f}")
    print(f"Position Error:  {err:.3f} m")
    print(f"Variance:        {variance:.6f}")
    print(f"ESS:             {pf.effective_sample_size():.2f}")
    print()

# =============================================================================
# --- CONFIGURATION FOR EXPERIMENTS ---
# =============================================================================
if __name__ == "__main__":
    
    # ***************************************************************
    # *** TO RUN YOUR EXPERIMENTS, CHANGE THESE VALUES ***
    # ***************************************************************
    run_all_experiments()
    # This is your "Tuning" panel
    # Config_Baseline = {
    #     "NUM_PARTICLES": 100,        # --- Experiment with this (e.g., 20, 100, 1000)
    #     "SIM_STEPS": 100,
        
    #     # --- True Robot Noise (Keep this constant) ---
    #     "ROBOT_D_STD": 0.1,    # True motion noise
    #     "ROBOT_TH_STD": 0.05,  # True motion noise
    #     "ROBOT_R_STD": 0.5,    # True sensor noise
        
    #     # --- Filter's *Belief* (This is what you tune) ---
        
    #     # Experiment 1: Process Noise (Q)
    #     "FILTER_D_STD": 0.1,   # Set > ROBOT_D_STD (cautious) or < (overconfident)
    #     "FILTER_TH_STD": 0.05, # Set > ROBOT_TH_STD (cautious) or < (overconfident)
        
    #     # Experiment 2: Measurement Noise (R)
    #     "FILTER_R_STD": 0.5,   # Set > ROBOT_R_STD (skeptical) or < (trusting)
    # }

    # # Run the simulation with the chosen configuration
    # run_simulation(Config_Baseline)