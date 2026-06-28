# Cart-Pole Reinforcement Learning Results

## Question 1: Convergence Trials

**How many trials (how many times did the pole fall over or the cart fall off) did it take before the algorithm converged?**

### Answer
The algorithm converged after **126 trials**.

### Explanation
- The reinforcement learning agent experienced 126 failures (pole falling or cart going out of bounds)
- At trial 126, the value iteration converged within a single iteration for the 20th consecutive time (as defined by `NO_LEARNING_THRESHOLD = 20`)
- This convergence criterion indicates that the estimated MDP model had stabilized and no significant learning was occurring, so the simulation terminated
- The learning procedure successfully balanced the pole with an increasingly stable policy over these 126 trials

## Question 2: Learning Curve Plot

The learning curve plot has been saved as `learning_curve.png` in this directory.

### Plot Details
- **X-axis**: Failure number (trial number)
- **Y-axis**: Number of time steps until failure (on logarithmic scale)
- **Black line**: Raw learning data
- **Red dashed line**: Smoothed learning curve (30-step moving average)

The plot shows the agent's improvement over time:
- Early trials: short balance times (pole falls quickly)
- Later trials: progressively longer balance times (agent learns better control)
- By trial 126: the curve plateaus, indicating convergence
