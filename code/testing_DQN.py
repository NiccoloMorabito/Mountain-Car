import gym
import numpy as np
from keras import models
import os
import matplotlib.pylab as plt
import tensorflow as tf
tf.keras.utils.disable_interactive_logging()

MODELS_FOLDER = 'all_models/'
NUM_EPISODES = 50

env = gym.make('MountainCar-v0')

model_to_avg_reward = dict()

for model_name in os.listdir(MODELS_FOLDER):
    print(f"using model {model_name}:")
    model_number = int(model_name.split("InEp")[1][:-3])
    model_path = os.path.join(MODELS_FOLDER, model_name)
    model=models.load_model(model_path)

    rewards = list()
    
    for episode in range(NUM_EPISODES):

        current_state = env.reset()
        total_reward = 0

        for t in range(200):
            action = np.argmax(model.predict(current_state[np.newaxis])[0])

            new_state, reward, done, info = env.step(action)

            current_state = new_state

            total_reward += reward

            if done:
                break
        
        rewards.append(total_reward)
    
    avg_reward = - np.mean(rewards)
    #print(f"The average score on {NUM_EPISODES} episodes is: {avg_reward}")
    print(f"{model_number} -> {avg_reward}")
    model_to_avg_reward[model_number] = avg_reward

myList = sorted(model_to_avg_reward.items())
x, y = zip(*myList) 
plt.plot(x, y)
plt.savefig('porcoddio.png')



env.close()