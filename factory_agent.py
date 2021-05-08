### Import packages ###
from factory_simulation import Factory_Simulation

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np
import random
import pickle
import simpy

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

import warnings
warnings.filterwarnings("ignore")

### Define model parameters ###

# Set whether to display plots on screen (slows model)
display_live_plot = True

# Simulation duration
simulation_duration = 60
# Training episodes
training_episodes = 10000
# Time step between actions
time_step = 1

# Discount rate of future rewards
GAMMA = 0.99
# Learing rate for neural network
LEARNING_RATE = 0.005
# Maximum number of game steps (state, action, reward, next state) to keep
MEMORY_SIZE = 1000000
# Sample batch size for policy network update
BATCH_SIZE = 8
# Number of game steps to play before starting training (all random actions)
REPLAY_START_SIZE = simulation_duration * 100

# Number of steps between policy -> target network update
SYNC_TARGET_STEPS = simulation_duration
# Exploration rate (episolon) is probability of choosign a random action
EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.005
# Reduction in epsilon with each game step
EXPLORATION_DECAY = 0.9995

# Term for plot adjustion (originally implemented to account for negative default rewards)
adjust_plot_factor = 0

### Define DQN (Deep Q Network) class, used for both policy and target nets ###
# build and structured according to https://github.com/MichaelAllen1966/learninghospital

class DQN(nn.Module):
    """Deep Q Network. Used for both policy (action) and target (Q) networks."""

    def __init__(self, observation_space, action_space, neurons_per_layer=48):
        """Constructor method. Set up neural nets."""

        # Set starting exploration rate
        self.exploration_rate = EXPLORATION_MAX
        
        # Set up action space (choice of possible actions)
        self.action_space = action_space
              
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, action_space))
        
    def act(self, state):
        """Act either randomly or by redicting action that gives max Q"""
        
        # Act randomly if random number < exploration rate
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_space)
            
        else:
            # Otherwise get predicted Q values of actions
            q_values = self.net(torch.FloatTensor(state))
            # Get index of action with best Q
            action = np.argmax(q_values.detach().numpy()[0])
        
        return  action
        
    def forward(self, x):
        """Forward pass through network"""
        return self.net(x)

### Define DDQN (Deep Q Network) class, used for both policy and target nets ###
# Please not that the agent uses the ddqn and not the previous dqn
class DDQN(nn.Module):
    """Duelling Deep Q Network. Used for both policy (action) and target (Q) networks."""

    def __init__(self, observation_space, action_space, neurons_per_layer=48):
        """Constructor method. Set up neural nets."""

        # Set starting exploration rate
        self.exploration_rate = EXPLORATION_MAX
        
        # Set up action space (choice of possible actions)
        self.action_space = action_space
              
        # First layers will be common to both Advantage and value
        super(DDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(observation_space, neurons_per_layer),
            nn.ReLU())
        
        # Advantage has same number of outputs as the action space
        self.advantage = nn.Sequential(
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, action_space))
        
        # State value has only one output (one value per state)
        self.value = nn.Sequential(
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, 1))        
        
    def act(self, state):
        """Act either randomly or by redicting action that gives max Q"""
        
        # Act randomly if random number < exploration rate
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_space)
            
        else:
            # Otherwise get predicted Q values of actions
            q_values = self.forward(torch.FloatTensor(state))
            # Get index of action with best Q
            action = np.argmax(q_values.detach().numpy()[0])
        
        return  action
        
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        action_q = value + advantage - advantage.mean()
        return action_q

### Define policy net training function 
def optimize(policy_net, target_net, memory, run):
    """ Update  model by sampling from memory. Uses policy network to predict best action (best Q).
    Uses target network to provide target of Q for the selected next action. """
    # Do not try to train model if memory is less than reqired batch size
    if len(memory) < BATCH_SIZE:
        return    

    # Reduce exploration rate (exploration rate is stored in policy net)
    policy_net.exploration_rate *= EXPLORATION_DECAY

    # Reduce exploration min periodically (used for testing)
    # policy_net.exploration_rate = max(EXPLORATION_MIN-(0.1*(run//50)), policy_net.exploration_rate)

    # Execute the last n runs with exploration rate of 0
    if (training_episodes-10) < run:
        policy_net.exploration_rate = max(0, policy_net.exploration_rate)
    else:
        policy_net.exploration_rate = max(EXPLORATION_MIN, policy_net.exploration_rate)

    # Sample a random batch from memory
    batch = random.sample(memory, BATCH_SIZE)
    for state, action, reward, state_next, terminal in batch:
        
        state_action_values = policy_net(torch.FloatTensor(state))
        
        # Get target Q for policy net update
       
        if not terminal:
            # For non-terminal actions get Q from policy net
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach next state values from gradients to prevent updates
            expected_state_action_values = expected_state_action_values.detach()
            # Get next state action with best Q from the policy net (double DQN)
            policy_next_state_values = policy_net(torch.FloatTensor(state_next))
            policy_next_state_values = policy_next_state_values.detach()
            best_action = np.argmax(policy_next_state_values[0].numpy())
            # Get target net next state
            next_state_action_values = target_net(torch.FloatTensor(state_next))
            # Use detach again to prevent target net gradients being updated
            next_state_action_values = next_state_action_values.detach()
            best_next_q = next_state_action_values[0][best_action].numpy()
            updated_q = reward + (GAMMA * best_next_q)      
            expected_state_action_values[0][action] = updated_q
        else:
            # For termal actions Q = reward (-1)
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach values from gradients to prevent gradient update
            expected_state_action_values = expected_state_action_values.detach()
            # Set Q for all actions to reward (-1)
            expected_state_action_values[0] = reward
 
        # Set net to training mode
        policy_net.train()
        # Reset net gradients
        policy_net.optimizer.zero_grad()  
        # calculate loss
        loss_v = nn.MSELoss()(state_action_values, expected_state_action_values)
        # Backpropogate loss
        loss_v.backward()
        # Update network gradients
        policy_net.optimizer.step()  

    return

### Define memory class ###

class Memory():
    """ Replay memory used to train model. Limited length memory (using deque, double ended queue from collections).
    - When memory full deque replaces oldest data with newest. Holds, state, action, reward, next state, and episode done. """
    
    def __init__(self):
        """Constructor method to initialise replay memory"""
        self.memory = deque(maxlen=MEMORY_SIZE)

    def remember(self, state, action, reward, next_state, done):
        """state/action/reward/next_state/done"""
        self.memory.append((state, action, reward, next_state, done))

### Define result plotting function ###

# Set up chart (ax1 and ax2 share x-axis to combine two plots on one graph)
fig = plt.figure(figsize=(16,9))
# Plot exploration rate of all runs
ax1 = fig.add_subplot(121)
# Plot the total rewards of all runs
ax2 = ax1.twinx()
# Plot the decisions made by the agent 
ax3 = fig.add_subplot(3,2,6) #ax3.twinx()
# Plot the accumulated rewards of the current run 
ax4 = fig.add_subplot(3,2,(2,4), sharex=ax3)

# Set default size for all texts
size = 12

def plot(run_results, run_explorations, run_total_rewards, episode_details, state, best_run_number, best_run_reward):

    ### Ax 1: Plot exploration rate of all runs ###
    
    # Clear previous plot on ax 1
    ax1.clear()
    # Set labels
    ax1.set_xlabel('Number of simulation runs', size=size)
    ax1.set_ylabel('Exploration rate', color='darkorchid', size=size)
    ax1.set_ylim(0, 1.1)
    # Plot data
    ax1.plot(run_results, run_explorations, label='Exploration rate', color='darkorchid', linestyle='--')

    ### Ax 2: Total reward of all runs ###

    # Clear previous plot on ax 2
    ax2.clear()
    # Set labels
    ax2.set_ylabel('Total reward of each run (adjusted for negative reward)', color='forestgreen', size=size)
    # Move axis to the left 
    #ax2.spines["left"].set_position(("axes", -0.15))
    #ax2.spines["left"].set_visible(True)
    #ax2.yaxis.set_label_position('left')
    #ax2.yaxis.set_ticks_position('left')
    # Set title
    ax2.set_title('Exploration rate and Total reward (for all {} runs)'.format(len(run_results)), size=size)
    # Adjust for negative reward training term (-20 * 480)
    run_total_rewards = [r+simulation_duration*adjust_plot_factor for r in run_total_rewards]  
    # Plot data
    ax2.plot(run_results, run_total_rewards, label='Reward', color='forestgreen', alpha=0.5)
    # Calculate and plot average rewards 
    if len(run_results) < 250:
        n=10
    else:
        n=25
    # Calculate average reward for each n simulation run
    avg_rewards = [ sum(run_total_rewards[i:i+n])/n for i in range(0, len(run_total_rewards), n) ]
    # Plot data
    ax2.plot(range((n//2), len(avg_rewards)*n, n)[:-1], avg_rewards[:-1], color='black')

    # Add grid
    ax2.grid()
    # Add a custom legend
    legend_lines = [Line2D([0], [0], color='darkorchid', linestyle='--', lw=1), Line2D([0], [0], color='forestgreen'), Line2D([0], [0], color='Black', lw=1)]
    ax2.legend(legend_lines, ['Exploration rate', 'Total reward', 'Total reward (avg.)'], loc=4)    

    ### Ax 3: Plot the decisions made by the agent as Scatter plot ###

    # Clear previous plot on ax 3
    ax3.clear()
    # Set labels
    ax3.set_ylabel('Selected action', size=size)
    ax3.set_xlabel('Simulation time', size=size)
    # Limit and rename labels
    ax3.set_ylim(-0.5,2.5)
    ax3.set_yticks([0,1,2])
    ax3.set_yticklabels(['A', 'B', '0'], size=18)
    ax3.get_yticklabels()[0].set_color('royalblue')
    ax3.get_yticklabels()[1].set_color('firebrick')
    ax3.get_yticklabels()[2].set_color('gold')
    # Move label to the right for readablility
    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()
    # Prepare actions and get color
    actions = episode_details['actions'] 
    marker_color = [ 'royalblue' if v==0 else 'firebrick' if v==1 else 'gold' for v in actions ]
    # Plot data 
    ax3.scatter(range(len(actions)), actions, label='actions', color=marker_color)
    # Add grid
    ax3.xaxis.grid()
    
    ### Ax 4: Plot the accumulated rewards of the current run ###
    
    # Adjust rewards for negative reward term (-20)
    rewards = [e+adjust_plot_factor for e in episode_details['rewards']]
    # Accumulate rewards for sum up to timestep 
    rewards_step = [ sum(rewards[:i]) for i in range(len(rewards))]
    # Clear plot of previous run
    ax4.clear()
    # Set title
    ax4.set_title('Results of run {r}: total_A={a}, total_B={b} and total_reward={t}'.format(r=len(run_results), a=state[0][-2], b=state[0][-1], t=sum(rewards)), size=size)
    # Set labels
    ax4.set_ylabel('Adjusted accumulated reward', size=size)    
    # Set limit 
    ax4.set_xlim(0, simulation_duration)
    # Plot data
    ax4.plot(rewards_step, label='reward', color='black')
    # Add best run for reference
    ax4.axhline(best_run_reward, color='silver', linestyle=':')
    # Move label to the right
    ax4.yaxis.set_label_position('right')
    ax4.yaxis.tick_right()
    # Add grid
    ax4.grid()
    # Add a custom legend
    legend_lines = [Line2D([0], [0], color='Black', lw=1), Line2D([0], [0], color='silver', linestyle=':', lw=1)]
    ax4.legend(legend_lines, ['Accumulated reward', 'Best acc. reward (run {})'.format(best_run_number)], loc=4)  

    ### Add layout and display plot ###
    #plt.tight_layout()
    plt.savefig('results/run_1/plots/image_{}'.format(run_results[-1]))
    plt.pause(0.001)

### Main programm ###

def run(training_episodes):
    """ Main program loop to execute simulation runs. """

    ### Set up game environment ###

    # Get simulation of factory
    factory_simulation = Factory_Simulation(
        simulation_duration=simulation_duration, 
        time_step=time_step)

    # Get number of observations returned for state
    observation_space = factory_simulation.observation_size

    # Get number of actions possible
    action_space = factory_simulation.action_size

    ### Set up policy and target nets ###

    # Set up policy and target neural nets
    policy_net = DDQN(observation_space, action_space)
    target_net = DDQN(observation_space, action_space)
    
    # Set loss function and optimizer
    policy_net.optimizer = optim.Adam(params=policy_net.parameters(), lr=LEARNING_RATE)
    
    # Copy weights from policy_net to target
    target_net.load_state_dict(policy_net.state_dict())
    
    # Set target net to eval rather than training mode
    # We do not train target net - ot is copied from policy net at intervals
    target_net.eval()

    ### Set up memory ###

    # Set up memory
    memory = Memory()    

    ### Set up and start training loop ###

    # Set up run counter and learning loop    
    run = 0
    all_steps = 0
    continue_learning = True

    # Set up tracker for best run
    best_run_number = 0
    best_run_reward = 0

    # Set up list for results
    run_results = []
    run_explorations = []
    run_total_rewards = []

    while continue_learning:

        ### Play episode ###

        # Increment run (episode) counter
        run += 1

        ### Reset game ###

        # Reset game environment and get first state observations
        state = factory_simulation.reset()  

        # Trackers for state
        actions = []
        rewards = []

        # Reset total reward
        total_reward = 0

        # Reshape state into 2D array with state obsverations as first 'row'
        state = np.reshape(state, [1, observation_space])

        # Continue loop until episode complete
        while True:

            ### Game episode loop ###

            # Get action to take (se eval mode to avoid dropout layers)
            policy_net.eval()
            action = policy_net.act(state)
            # action = random.randint(0,2) # to test random runs

            ### Play action (get S', R, T) ### 

            # Act 
            state_next, reward, terminal, info = factory_simulation.step(action)
            total_reward += reward

            # Update trackers
            actions.append(action)
            rewards.append(reward)

            # Reshape state into 2D array with state obsverations as first 'row'
            state_next = np.reshape(state_next, [1, observation_space])

            # Add S, A, R, S', T to memory
            memory.remember(state, action, reward, state_next, terminal)
            
            # Update state
            state = state_next

            ### Check for end of episode ###

            if terminal:
                
                print('Run {} done.'.format(run))
                
                # Get the exploration rate
                exploration = policy_net.exploration_rate
                
                # Add results to result lists
                run_results.append(run)
                run_explorations.append(exploration)
                run_total_rewards.append(total_reward)

                # Change if best run so far
                if total_reward+adjust_plot_factor*simulation_duration > best_run_reward:
                    best_run_reward = total_reward+adjust_plot_factor*simulation_duration
                    best_run_number = run

                # Check for end of learning 
                if run == training_episodes:
                    continue_learning = False
                
                # End episode loop
                break

            ### Update policy net ###
            
            # Avoid training model if memory is not of sufficient length
            if len(memory.memory) > REPLAY_START_SIZE:
        
                # Update policy net
                optimize(policy_net, target_net, memory.memory, run) 

                # Update target net periodically and use load_state_dict method to copy weights from policy net       
                if all_steps % SYNC_TARGET_STEPS == 0:
                    target_net.load_state_dict(policy_net.state_dict()) 
        
        # Save episode details to DF
        episode_details = pd.DataFrame()
        episode_details['steps'] = range(len(rewards))
        episode_details['rewards'] = rewards
        episode_details['actions'] = actions 

        # Plot the episode results when the run is completed 
        if display_live_plot: 
            # Update result plot, using the new run
            plot(run_results, run_explorations, run_total_rewards, episode_details, state, best_run_number, best_run_reward)

        # Simply save result list to pickle 
        with open('results/run_1/data/results_1_run_{}.pkl'.format(run_results[-1]), 'wb') as f:
            pickle.dump([run_results, run_explorations, run_total_rewards, episode_details, state, best_run_number, best_run_reward], f)

    # Return details of last episode
    return episode_details

# Adjust setupt_time in Factory_Simulation for comparison 
# results/run_1/ --> for setup_time = 0
# results/run_2/ --> for setup_time = 5
# results/run_3/ --> for setup_time = 10

# Execute training run and save last run
last_run = run(training_episodes)

# Return final plot 
if display_live_plot:
    plt.show()
