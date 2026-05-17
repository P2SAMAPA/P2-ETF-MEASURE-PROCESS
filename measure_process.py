import numpy as np

def fleming_viot_particle_process(historical_returns, n_particles=100, mutation_std=0.01, selection_intensity=1.0):
    """
    Simulate a Fleming‑Viot superprocess approximated by a particle system.
    historical_returns: array of daily returns over a window (length T)
    Returns: predicted next return (weighted mean of final particle distribution)
    """
    T = len(historical_returns)
    if T < 2:
        return 0.0
    # Initialise particles with bootstrap sample from historical returns
    particles = np.random.choice(historical_returns, size=n_particles, replace=True)
    # Evolve over time steps (each step corresponds to one day)
    for t in range(T):
        # Mutation step: add Gaussian noise
        particles += np.random.normal(0, mutation_std, size=n_particles)
        # Selection step: resample particles with probability proportional to fitness
        # Fitness = exp(selection_intensity * return) – encourages high returns
        # Use current return at time t as reference? Actually, fitness should be based on the
        # observed return at that time? The Fleming-Viot process uses a "selection" term that
        # favours particles with higher growth. We'll use the observed daily return as a target
        # to weight particles.
        observed_return = historical_returns[t]
        # Compute fitness of each particle: closeness to observed return? That would be
        # a different interpretation. In evolutionary biology, fitness is the reproductive success.
        # Here we want particles that represent high returns to be weighted more.
        # We'll use a simple: fitness = exp(selection_intensity * particle_value)
        # Because we want the measure to shift toward higher returns.
        fitness = np.exp(selection_intensity * particles)
        # Avoid division by zero
        fitness_sum = np.sum(fitness)
        if fitness_sum == 0:
            fitness = np.ones(n_particles) / n_particles
        else:
            fitness = fitness / fitness_sum
        # Resample particles according to fitness
        indices = np.random.choice(n_particles, size=n_particles, p=fitness, replace=True)
        particles = particles[indices]
    # Final prediction: weighted mean of particles (fitness weighting already accounted in resampling)
    # But we can also compute mean of the final particles (since they are equally weighted after resampling)
    return np.mean(particles)

def measure_process_prediction(returns_series, window, n_particles=100, mutation_std=0.01, selection_intensity=1.0):
    """
    For a single ETF, use the last `window` days of returns to simulate the measure-valued process,
    and predict the next day's return.
    """
    if len(returns_series) < window:
        return 0.0
    hist = returns_series.iloc[-window:].dropna().values
    if len(hist) < 10:
        return 0.0
    pred = fleming_viot_particle_process(hist, n_particles, mutation_std, selection_intensity)
    # The prediction may be very small; we can return it directly.
    return pred
