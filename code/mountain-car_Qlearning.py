import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym
import imageio

#Initializations
number_states = 40 #number of states used by agent
max_episodes = 5000 
initial_learning_rate = 1.0
min_learning_rate = 0.005   
max_step = 270 #max steps each episode will last

#Initialise parameters for Q-Learning
#epsilon = 0 means no random action thus no exploration
#epsilon = 0

#epsilon = 0.5 we take a random action with prob 0.5
#epsilon = 0.5

#Start with moderate epsilon and slowly decrease after each episode with decay rate
epsilon = 0.3

#Set the decay rate for epsilon
decay_rate = 0.01
gamma = 1 #discount factor


#Function to map observation to state
def observation_to_state(environment, observation):
    environment_low = environment.observation_space.low
    environment_high = environment.observation_space.high
    environment_dx = (environment_high - environment_low) / number_states

    #Set position and velocity
    p = int((observation[0] - environment_low[0])/environment_dx[0])
    v = int((observation[1] - environment_low[1])/environment_dx[1])

    return p, v

#Function to simulate episode with a given policy
def simulate_episode(environment, policy=None, render=False, save_gif=False):
    observation= environment.reset()
    total_reward = 0
    step_count = 0

    #Create a list to store the frames of the render
    frames = []

    for _ in range(max_step):
        #Save the current frame of the environment
        #frame = environment.render(mode='rgb_array')
        #frames.append(frame)
        if policy is None:
            action = environment.action_space.sample()
        else:
            p,v = observation_to_state(environment, observation)
            action = policy[p][v]
        if render:
            environment.render()
        #Get observation, reward and done after each step
        observation, reward, done, _ = environment.step(action)
        total_reward += gamma ** step_count * reward
        step_count += 1
        if done:
            break
    if save_gif:
        imageio.mimsave('/Users/davide/Desktop/Rl_Project/mountain-car_qlearning.gif', frames, fps=30)

    return total_reward

if __name__ == '__main__':
    #Setting up mountain car environment
    environment_name = "MountainCar-v0"
    env = gym.make(environment_name)
    env.seed(0)
    np.random.seed(0)

    #Print the observation space
    print("Observation space:", env.observation_space)

    #Print the action space
    print("Action space:", env.action_space)
    
    #Create Q-Table with zeros representing the 3 actions in the action-space
    q_table = np.zeros((number_states, number_states, 3))
    #print(q_table)

    #Create a list to store the rewards for each episode
    rewards = []
    # Create a list to store the final position of the car at each episode
    car_positions = []

    print("Running episodes")
    #Training for maximum episodes
    for i in range(max_episodes):
        observation = env.reset()
        total_reward = 0

        #Learning rate is decreased at each step
        alpha = max(min_learning_rate, initial_learning_rate * (0.85 ** (i//100)))
        #Each episode is max_step long
        for j in range(max_step):
            p, v = observation_to_state(env, observation)
            #Select an action
            if np.random.uniform(0, 1) < epsilon:
                #Get random action
                action = np.random.choice(env.action_space.n)
            else:
                logits = q_table[p][v]
                #Calculate the exponential of all elements in the input array.
                logits_exp = np.exp(logits)
                #Calculate the probabilities
                probabilities = logits_exp / np.sum(logits_exp)
                #Get random action
                action = np.random.choice(env.action_space.n, p=probabilities)
            
            #Get observation, reward and done after each step
            observation, reward, done, _ = env.step(action)

            total_reward += reward
            #Update Q-table
            p_, v_ = observation_to_state(env, observation)
            
            #Update according to Bellmann equation: Q(s, a) = reward + gamma * max(Q(s', a'))
            q_table[p][v][action] = q_table[p][v][action] + alpha * (reward + gamma *  np.max(q_table[p_][v_]) - q_table[p][v][action])
            if done:
                break

        #Decrease epsilon using an exponential decay function
        epsilon = epsilon * decay_rate

        #Append the total reward for this episode to the list of rewards
        rewards.append(total_reward)
        #Print every 100 episode the total reward
        #if i % 100 == 0:
            #print("Episode number: %d - Total Reward: %d." %(i+1, total_reward))
    
    #Smooth the curve by taking the rolling average over 100 episodes
    rolling_window_size = 100
    rewards_smooth = pd.Series(rewards).rolling(rolling_window_size).mean()

    print(rewards)

    #Plot the rewards
    #plt.plot(rewards)
    #plt.xlabel("Episode")
    #plt.ylabel("Total Reward")
    #plt.show()

    #Plot the smoothed curve
    #plt.plot(rewards_smooth)
    #plt.xlabel("Episode")
    #plt.ylabel("Total Reward (Smoothed)")
    #plt.show()
    
    #Derive policy from Q-table
    solution_policy = np.argmax(q_table, axis=2)
    #Test policy
    solution_policy_scores = [simulate_episode(env, solution_policy, False, False) for _ in range(50)]

    print("Episodes avg score: ", np.mean(solution_policy_scores))
    simulate_episode(env, solution_policy, True, False)