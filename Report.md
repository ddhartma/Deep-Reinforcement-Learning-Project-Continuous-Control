[image1]: assets/trained_agent.gif
[image2]: assets/2.png

# Deep Reinforcement Learning Project - Continuous Control - Implementation Report

## Content
- [Implementation - Continuous_Control.ipynb](#impl_notebook_train)
- [Implementation - ddpg_agent.py](#impl_agent)
- [Implementation - model.py](#impl_model)
- [Implementation - Continuous_Control_Trained_Agent.ipynb](#impl_notebook_trained_agent)
- [Ideas for future work](#ideas_future)

## Implementation - Continuous_Control.ipynb <a name="impl_notebook_train"></a>
- Open jupyter notebook file ```Continuous_Control.ipynb```
    ### Import important libraries
    - modul ***unityagents*** provides the Unity Environment. This modul is part of requirements.txt. Check the README.md file for detailed setup instructions.
    - modul **ddpg_agent** contains the implementation of an DDPG agent. Check the description of **ddpg_agent.py** for further details. 
    ```
    import gym
    import random
    import torch
    import numpy as np
    from collections import deque
    import matplotlib.pyplot as plt
    %matplotlib inline

    from unityagents import UnityEnvironment
    from ddpg_agent import Agent
    ```
    ### Instantiate the Environment
    - Load the UnityEnvironment and store it in **env**. Here I have chosen version 1 (a single agent).Check out README for further information
    - Environments contain brains which are responsible for deciding the actions of their associated agents.
    ```
    env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe', no_graphics=True, worker_id=1)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    ```
    ### Collect information about the environment
    ```
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print()
    print('The state for the first agent looks like: \n', states[0])
    ```
    - **Number of agents (states.shape[0]):** 20
    - **Size of each action (action_size):** 4
    - **Agent observes a state with length (states.shape[1]):** 33
    - **The state for the first agent looks like (states[0]):**
    ``` 
    [ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
    -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
    1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00 -6.30408478e+00 -1.00000000e+00
    -4.92529202e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
    -5.33014059e-01]
    ```
    ### Take Random Actions in The Environment - Understand types and shapes of variables
    - **states** - (numpy array) with shape (1,33), a vector with 33 float values, contains the agent's position, rotation, velocity, angular-velocity of the arm. Given this information, the agent has to learn how to best select actions.
    - **actions** - (numpy array) with shape (1,4), a vector with 4 float values
    - **env_info** - (unityagent instance)
    - **next_states** - (numpy array) next state (here chosen by random action), same config as states
    - **rewards** - (list) of float values, here: len(rewards) is 1 (one agent), a reward of +0.1 is provided for for each step that the agent's hand is in the goal location
    - **dones** - (list) of bool values, here: len(dones) is 1 (one agent), if True episode is over
    - **scores** - (numpy array) withshape (1,), cumulative reward, scoring after each action
    ```
    env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break

    print('states')
    print(states)
    print('states shape', states.shape)
    print('type', type(states))
    print()
    print('actions ', actions)
    print('actions shape', actions.shape)
    print('type', type(actions))
    print()
    print('env_info')
    print(env_info)
    print()
    print('next_states')
    print(next_states)
    print('next_states shape', next_states.shape)
    print('type', type(next_states))
    print()
    print('rewards')
    print(rewards)
    print('type', type(rewards))
    print()
    print('dones')
    print(dones)
    print('type', type(dones))
    print()
    print('scores')
    print(scores)
    print('scores shape', scores.shape)
    print('type', type(scores))
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    RESULTS:
    ------------
    states
    [[ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
    -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
    1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00 -5.14229965e+00 -1.00000000e+00
    6.12835693e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
    3.78988951e-01]
    ...
        x 20 agents
    ]
    states shape (20, 33)
    type <class 'numpy.ndarray'>

    actions  [[-1.         -1.         -0.61081709 -0.06546604] x 20 agents]
    actions shape (20, 4)
    type <class 'numpy.ndarray'>

    env_info
    <unityagents.brain.BrainInfo object at 0x0000023ECCC1CBA8>

    next_states
    [[ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
    -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
    1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00 -5.14229965e+00 -1.00000000e+00
    6.12835693e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
    3.78988951e-01]
    ...
        x 20 agents
    ]
    next_states shape (20, 33)
    type <class 'numpy.ndarray'>

    rewards
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    type <class 'list'>

    dones
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    type <class 'list'>

    scores
    [0.         0.         0.13       0.48999999 0.         0.
     0.         0.44999999 0.         0.         0.         0.63999999
     0.         0.         0.         0.24999999 0.         0.
     0.         0.        ]
    scores shape (20,)
    type <class 'numpy.ndarray'>
    Total score (averaged over agents) this episode: 0.09799999780952931
    ```
    ### Instantiate the Agent
    ```
    agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=42)

    RESULTS:
    ------------
    Training on  cpu

    ------- ACTOR ------
    self.actor_local Actor(
    (fc1): Linear(in_features=33, out_features=256, bias=True)
    (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc2): Linear(in_features=256, out_features=128, bias=True)
    (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc3): Linear(in_features=128, out_features=4, bias=True)
    )

    self.actor_target Actor(
    (fc1): Linear(in_features=33, out_features=256, bias=True)
    (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc2): Linear(in_features=256, out_features=128, bias=True)
    (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc3): Linear(in_features=128, out_features=4, bias=True)
    )

    self.actor_optimizer Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        eps: 1e-08
        lr: 0.0001
        weight_decay: 0
    )

    ------- CRITIC ------
    self.critic_local Critic(
    (fcs1): Linear(in_features=33, out_features=256, bias=True)
    (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc2): Linear(in_features=260, out_features=128, bias=True)
    (fc3): Linear(in_features=128, out_features=1, bias=True)
    )

    self.critic_target Critic(
    (fcs1): Linear(in_features=33, out_features=256, bias=True)
    (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc2): Linear(in_features=260, out_features=128, bias=True)
    (fc3): Linear(in_features=128, out_features=1, bias=True)
    )

    self.critic_optimizer Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        eps: 1e-08
        lr: 0.0001
        weight_decay: 0
    )
    ```
   
    ### Train an Agent
    - the code basis for **ddpg() function** has been taken from the file **DDPG.ipynb** in the Udacity repo [ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)
    - function which implements the agent training
    - 2 for loops: outer loop --> loop over episodes and inner loop --> loop over timesteps per episode (TD learning algorithm)
        - for the actual episode:
            - reset the environment
            - get the current state
            - reset the agent
            - initialize the score 
            - for the actual time step:
                - return actions for current state and policy
                - send the action to the environment
                - get the next state
                - get the reward 
                - see if episode has finished
                - update the agent's knowledge, using the most recently sampled tuple (-->agent.step)
            - save most recent score
            - save torch models for actor and critic if average score >=30
        - return scores    
    ```
    def ddpg(n_episodes=1000, max_t=1000):
        """ Deep Deterministic Policy Gradient
        
            INPUTS: 
            ------------
                n_episodes - (int) maximum number of training episodes
                max_t - (int) maximum number of timesteps per episode
                
            OUTPUTS:
            ------------
                scores - (list) list of score values for each episode
        """ 
        best_score = -np.inf                                       # initialize best score as minus infinite 
        mean_scores = []                                           # list containing scores from each episode
        mean_score_window = deque(maxlen=10)                       # last 10 scores
        mean_scores_window_avg = []                                # list container to store the rolling avg of mean_score_window
        consec_episodes = 0                                        # counter for consecutive episodes with mean_scores_window_avg > 30
        for i_episode in range(1, n_episodes+1):                   # start for loop over episodes
            env_info = env.reset(train_mode=True)[brain_name]      # get/reset enviroment      
            states = env_info.vector_observations                  # get states values for all 20 agents from environment
            agent.reset()                                          # reset the agents' noise
            score = np.zeros(num_agents)                           # initialize episode score for all 20 agents
            start_time = time.time()                               # start episode timer
            for t in range(max_t):                                 # start for loop over episode's time steps
                actions = agent.act(states)                        # return actions for current states and policy
                env_info = env.step(actions)[brain_name]           # send all actions to the environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                
                # save experience to replay buffer,
                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    agent.step(state, action, reward, next_state, done, t)  

                states = next_states                                      # save next_sates as actual states
                score += rewards                                          # add up rewards of all 20 agents for this time step to episode score 
                
                if np.any(dones):                                         # exit loop if episode finished
                    break
            
            duration = time.time() - start_time                           # calculate elapsed time for actual episode
            mean_scores.append(np.mean(score))                            # save most recent score
            mean_score_window.append(mean_scores[-1])                     # add to consecutive scores
            mean_scores_window_avg.append(np.mean(mean_score_window))     # save the most recent consecutive score

            # if actual mean_score averaged over 20 agents and over 10 consecutive episodes is LOWER than 30 ... 
            # print scores, set consec_epsiodes to 0 if needed
            if mean_scores_window_avg[-1] < 30:
                print('\rEpisode {}\tDur: {:.1f} \tAverage Eps Score: {:.2f} \tMean Consec Score: {:.2f}'.format(i_episode, round(duration), mean_scores[-1], mean_scores_window_avg[-1]))
                if consec_episodes != 0:
                    consec_episodes = 0
            
            # if actual mean_score averaged over 20 agents and over 10 consecutive episodes is HIGHER than 30 ... 
            # print scores, increase consec_epsiodes, update pytorch model state_dicts and save them
            # if consec_episodes == 100 --> task is successfully completed.
            if mean_scores_window_avg[-1] >= 30:
                consec_episodes += 1
                print('\rEpisode {}\tDur: {:.1f} \tAverage Eps Score: {:.2f} \tMean Consec Score: {:.2f} \tConsec Eps: {:.2f}'.format(i_episode, round(duration), mean_scores[-1], mean_scores_window_avg[-1], int(consec_episodes)))

                
                if mean_scores_window_avg[-1] > best_score:
                    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
                    best_score = mean_scores_window_avg[-1]
                
                if consec_episodes == 100:
                    print('\nEnvironment SOLVED : \tMoving Average ={:.1f} over last {} episodes'.format(mean_scores_window_avg[-1], consec_episodes))            
                    break
        
        return mean_scores, mean_scores_window_avg


    mean_scores, mean_scores_window_avg = ddpg()
    ```
    ### Plot the cumulated scores as a function of episodes
    ```
    # Plot mean_scores and mean_scores_window_avg (moving average)
    target = [30] * len(mean_scores) # Trace a line indicating the target value
    fig = plt.figure(figsize=(18,8))
    fig.suptitle('Plot of the rewards', fontsize='xx-large')

    ax = fig.add_subplot(111)
    ax.plot(mean_scores, label='Score', color='Blue')
    ax.plot(mean_scores_window_avg, label='Moving Average',
            color='LightGreen', linewidth=3)
    ax.plot(target, linestyle='--', color='LightCoral', linewidth=1 )
    ax.text(0, 30, 'Target', color='LightCoral', fontsize='large')
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    ax.legend(fontsize='xx-large', loc='lower right')

    plt.show()
    ```
    ![image2]

    - **Score** has been averaged over all 20 agents.
    - **Moving Average** is the rolling average with a roll length of 10, i.e. the **Moving Average** is calculated from 10 consecutive **Score** values.   
    - After **21 episodes** the moving average was always **greater than 30**. Therefore the task (average score over all 20 agents AND higher than 30 for at least 100 episodes) has been **successfully completed at episode 121**.

    ### Close the environment
    ```
    env.close()
    ```

## Implementation - ddpg_agent.py <a name="impl_agent"></a>
- Open Python file ```notebooks_python/ddpg_agent.py```
- The code basis for **ddpg_agent.py** has been taken from the file **ddpg_agent.py** in the Udacity repo [ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)
    ### Load important libraries
    ```
    import numpy as np
    import random
    import copy
    from collections import namedtuple, deque

    from model import Actor, Critic

    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    ```
    ### Hyperparameters
    ```
    BUFFER_SIZE = int(1e6)  # replay buffer size
    BATCH_SIZE = 128        # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR_ACTOR = 1e-4         # learning rate of the actor
    LR_CRITIC = 1e-4        # learning rate of the critic
    WEIGHT_DECAY = 0        # L2 weight decay
    NOISE_DECAY = 0.999     # NOISE Decay (not needed)
    UPDATE_EVERY = 20       # Update experience tuple every <UPDATE_EVERY> time steps
    NUM_UPDATES = 10        # Number of updates --> call of learn function every <NUM_UPDATES>
    NOISE_SIGMA = 0.05      # Diffusion parameter for Ornstein-Uhlenbeck noise, weight parameter for adding random

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ```
    - A **BUFFER_SIZE** in the range 1e6 to 1e5 seems to be efficient in order to sample experiences from the replay buffer.  
    - Different ***BATCH_SIZE** values for minibatches were tested. 128 seems to be a good compromise with regard to training speed (the higher the BATCH_SIZE the slower the training progress) and training performance.
    - The discount factor **GAMMA** has been set to 0.99 as a standard value. Other values could be checked in future approaches. The larger the discount rate is, the more the agent cares about the distant future.
    - In order to ensure a **soft update** of the target networks by th local (regular) networks the **TAU** parameter has been limited to low values. Higher values would lead to more aggressive target updates, i.e. the local network weights would have more updating power on the target network weights.
    - Different learning rates were tested during training, both for the actor and critic networks. Higher learning rates could on the one hand increase training speed but on the other lead to update oscillations. Reasonable learning rates were found in the range between 1e-4 to 1e-3. However, the best results were found for **LR_ACTOR** = **LR_CRITIC** = 1e-4 voth for actor and critic.    
    - **UPDATE_EVERY** and **NUM_UPDATES**: It has been found out, that instead of updating the actor and critic networks 20 times at every timestep, a network update **10 times after every 20 timesteps** increased stability and accelareted convergence.
    - Lower values than 0.2 for **NOISE_SIGMA** (Diffusion parameter for Ornstein-Uhlenbeck noise, weight parameter for adding random to the kinetics) seem to be benficial with regard to stability and fast convergence.
    
    ### Main class 'Agent' to train an agent getting smart
    - **init** function:
        - design the DDPG-Network based on the numbers state_size=33 and action_size=4, see model.py for more information
        - 4 neural networks (2 for Actor, 2 for Critic)
            - self.actor_local
            - self.actor_target
            - self.critic_local
            - self.critic_target
        - use Adam optimizer (minibatch SGD with adaptive learning rate, momentum)
        - initialize Ornstein-Uhlenbeck noise  
        - initialize epsilon for epsilon-greedy policies
        - initialize Replaybuffer: for storing each experienced tuple in this buffer. This buffer helps to reduce correlation between consecutive SARS tuples, by random sampling of experiences from this buffer

    - **step** function:
        - save experience in replay memory **self.memory**
        - use random sample from buffer to learn
        - structure of **experiences** (tuple of torch tensors):
            ```
            (
                tensor([[33x floats for state], x self.batch_size]),
                tensor([[4x int for action], x self.batch_size]),
                tensor([[1x float for reward], x self.batch_size]),
                tensor([[33x floats for next_state], x self.batch_size]),
                tensor([[1x int for done], x self.batch_size])
            )
            ```
        - if ReplayBuffer is larger than BATCH_SIZE actual time_step == UPDATE_EVERY, then sample from Replaybuffer 
        - repeat the learn/update process NUM_UPDATES times
        
    
    - **act** function:
        - returns a **clipped_action** for given state and current policy
        - convert **state** from numpy array to torch tensor
        - use agent **state** as input for **actor_local**
        - **actor_local** is used to find the best action
        - set **actor_local model** to evaluation mode (no optimizer step, no backpropagation)
        - get action values as Fixed Targets
        - set **actor_local model** back to train mode
        - optional: add Ornstein-Uhlenbeck noise to action
        - limit action to a **clipped_action**
        
    - **learn** function:
        - update policy and value parameters using given batch of experience tuples
        - Input: **experiences** (tuple of torch tensors):
            ```
            (
                tensor([[33x floats for states], x self.batch_size for minibatch]),
                tensor([[4x int for actions], x self.batch_size for minibatch]),
                tensor([[1x float for rewards], x self.batch_size for minibatch]),
                tensor([[33x floats for next_states], x self.batch_size for minibatch]),
                tensor([[1x bool for dones], x self.batch_size for minibatch])
            )
            ```
        - compute and minimize the loss for **actor** and **critic**
            1. **Update local critic**
                - get max predicted **Q_targets_next** (for next states) from **critic target model**
                    ```
                    Q_targets_next = self.critic_target(next_states, actions_next)

                    RESULT structure for Q_targets_next:
                    tensor([[1x float], x self.batch_size])
                    ```
                - compute **Q_targets** for current states 
                    ```
                    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

                    RESULT structure for Q_targets:
                    tensor([[1x float], x self.batch_size])
                    ```
                - get expected Q values **Q_expected** from **critic local model**
                    ```
                    Q_expected = self.critic_local(states, actions)

                    RESULT structure for Q_expected:
                    tensor([[1x float], x self.batch_size])
                    ```
                - compute loss **critic_loss**
                    ```
                    F.mse_loss(Q_expected, Q_targets)

                    RESULT structure for critic_loss:
                    tensor(1x float)
                    ```
                - minimize loss
                    ```
                    self.critic_optimizer.zero_grad() 
                    critic_loss.backward()
                    ```
                - update **local critic network**
                    ```
                    self.critic_optimizer.step()
                    ```
            
            2. **Update local actor**
                - Forward pass: Input states, output **actions_pred**
                    ```
                    actions_pred = self.actor_local(states)

                    RESULT structure for actions_pred:
                    tensor([[4x float], x self.batch_size for minibatch])
                    ```
                - Compute **actor_loss** via critic local
                    ```
                    actor_loss = -self.critic_local(states, actions_pred).mean()

                    RESULT structure for actor_loss:
                    tensor(1x float)
                    ```
                - minimize the loss
                    ```
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    ```
                - update **local actor network**
                    ```
                    self.actor_optimizer.step()
                    ```
            
            3. **Soft Updates of target networks**
                - Update target critic and actor
                    ```
                    self.soft_update(self.critic_local, self.critic_target, TAU)
                    self.soft_update(self.actor_local, self.actor_target, TAU)
                    ```

    - **soft_update** function:
        - soft update model parameters
        - Soft Update strategy consists of **slowly blending local (regular) network weights with target network weights** 
    ```
    class Agent():
        """ Interacts with and learns from the environment
        """

        def __init__(self, state_size, action_size, num_agents, random_seed, checkpoint_actor=None, checkpoint_critic=None):
            """Initialize an Agent object.

                INPUTS:
                ------------
                    state_size - (int) dimension of each state
                    action_size - (int) dimension of each action
                    random_seed - (int) random seed
                    checkpoint_actor -  (string) path to actor model parameters
                    checkpoint_critic - (string) path to critic model parameters

                OUTPUTS:
                ------------
                    No direct
            """
            print('Training on ', device)
            print()

            self.state_size = state_size
            self.action_size = action_size
            self.num_agents = num_agents
            self.seed = random.seed(random_seed)

            # Actor Network (w/ Target Network)
            self.actor_local = Actor(state_size, action_size, random_seed).to(device)
            self.actor_target = Actor(state_size, action_size, random_seed).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

            # Critic Network (w/ Target Network)
            self.critic_local = Critic(state_size, action_size, random_seed).to(device)
            self.critic_target = Critic(state_size, action_size, random_seed).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

            # Load model parameters
            if checkpoint_actor != None:
                self.actor_local.load_state_dict(torch.load(checkpoint_actor))
                self.actor_target.load_state_dict(torch.load(checkpoint_actor))
            if checkpoint_critic != None:
                self.critic_local.load_state_dict(torch.load(checkpoint_critic))
                self.critic_target.load_state_dict(torch.load(checkpoint_critic))

            # Noise process
            self.noise = OUNoise(action_size, random_seed)

            self.noise_decay = NOISE_DECAY

            # Replay memory
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

            self.t_step = 0

            print('------- ACTOR ------')
            print('self.actor_local',self.actor_local)
            print()
            print('self.actor_target',self.actor_target)
            print()
            print('self.actor_optimizer',self.actor_optimizer)
            print()
            print('------- CRITIC ------')
            print('self.critic_local',self.critic_local)
            print()
            print('self.critic_target',self.critic_target)
            print()
            print('self.critic_optimizer',self.critic_optimizer)
            print()


        def step(self, state, action, reward, next_state, done, timestep):
            """ Save experience in replay memory, and use random sample from buffer to learn.

                INPUTS:
                ------------
                    state - (numpy array) with shape (33,) state vector for the actual agent
                    action - (numpy array) with shape (4,) actual action values for the agent
                    reward - (float) actual reward
                    next_state - (numpy array) with shape (33,) next state vector for the actual agent
                    done - (bool) if True epsiode is finished for the agent
                    timestep ()

                OUTPUTS:
                ------------
                    No direct
            """


            # Save experience / reward
            self.memory.add(state, action, reward, next_state, done)
            

            # Learn, if enough samples are available in memory
            # Learn every <UPDATE_EVERY> time steps.
            # Repeat learning process <NUM_UPDATES> times
            if len(self.memory) > BATCH_SIZE and timestep % UPDATE_EVERY == 0:
                for _ in range(NUM_UPDATES):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
          
        def act(self, state, add_noise=True):
            """ Returns actions for given state as per current policy.

                INPUTS:
                ------------
                    state - (numpy array) with shape (1,33), a vector with 33 float values, contains the agent's position, rotation, velocity, angular-velocity of the arm. Given this information, the agent has to learn how to best select actions.
                    add_noise - (bool) if True add noise to chosen action based on Ornstein-Uhlenbeck

                OUTPUTS:
                ------------
                    clipped_action - (numpy array) with shape (1,4) each value of action is clipped withion (-1, 1)
            """
           
            state = torch.from_numpy(state).float().to(device)
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()
            self.actor_local.train()
            if add_noise:
                action += self.noise.sample() #* self.noise_decay
                #action *=  self.epsilon
                self.noise_decay *= self.noise_decay
            else:
                clipped_action =  action

            clipped_action = np.clip(action, -1, 1)

            #print('clipped_action')
            #print(clipped_action)
            #print('state shape', clipped_action.shape)
            #print('type', type(clipped_action))
            #print()

            return clipped_action

        def reset(self):
            self.noise.reset()

        def learn(self, experiences, gamma):
            """ Update policy and value parameters using given batch of experience tuples.
                Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
                where:
                    actor_target(state) -> action
                    critic_target(state, action) -> Q-value

                INPUTS:
                ------------
                    experiences - (Tuple[torch.Tensor]) tuple of (s, a, r, s', done) tuples
                    gamma - (float) discount factor

                OUTPUTS:
                ------------
                    No direct
            """
    
            states, actions, rewards, next_states, dones = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic_local(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)

            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local(states, actions_pred).mean()

            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

            # ---------------------------- update noise ---------------------------- #
            self.reset()

        def soft_update(self, local_model, target_model, tau):
            """ Soft update model parameters.
                θ_target = τ*θ_local + (1 - τ)*θ_target

                INPUTS:
                ------------
                    local_model - (PyTorch model) weights will be copied from
                    target_model - (PyTorch model) weights will be copied to
                    tau - (float) interpolation parameter

                OUTPUTS:
                ------------
                    No direct
            """

            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    ```
  
    ### ReplayBuffer class to reduce consecutive experience tuple correlations
    - **init** function: initialize a ReplayBuffer object
    - **add** function: add an experience tuple to self.memory object
    - **sample** function: 
        - create a random **experiences** sample (of type list) from **self.memory** ReplayBuffer instance with size **BATCH_SIZE**
        - size of ReplayBuffer (self.memory): BUFFER_SIZE
        - structure of experience sample: list of named tuples **Experience** with length **BATCH_SIZE**
            ```
            [
                Experience(
                            **state**=array([33 xfloat values]),
                            **action**=array([4 x float values]),
                            **reward**=1 x float,
                            **next_state=array([33 x float values])
                            ),
               
                ... x BATCH_SIZE
            ]              
            ```
        - return **states**, **actions**, **rewards**, **next_states** and **dones** each as torch tensors
        - structure of 
            - **states**: torch tensor with shape (BATCH_SIZE,33)
            - **actions**: torch tensor with shape (BATCH_SIZE,4)
            - **rewards**: torch tensor with shape (BATCH_SIZE,1)
            - **next_states**: torch tensor with shape (BATCH_SIZE,33)
            - **dones**: torch tensor with shape (BATCH_SIZE,1)
            ```
            states: tensor([[ <float value,> x 33 ], x BATCH_SIZE ]) 
            actions: tensor([[ <float value,> x 4 ], x BATCH_SIZE ]) 
            rewards: tensor([[ <float value> ], x BATCH_SIZE ]) 
            next_states: tensor([[ <float value,> x 33 ], x BATCH_SIZE ]) 
            dones: tensor([[ <bool value> ], x BATCH_SIZE ]) 
            ```
    - **__len__** function: return the current size of internal memory
    ```
    class ReplayBuffer:
        """ Fixed-size buffer to store experience tuples.
        """

        def __init__(self, action_size, buffer_size, batch_size, seed):
            """ Initialize a ReplayBuffer object.

                INPUTS:
                ------------
                    buffer_size - (int) maximum size of buffer
                    batch_size - (int)size of each training batch

                OUTPUTS:
                ------------
                    No direct
            """

            self.action_size = action_size
            self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
            self.batch_size = batch_size
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
            self.seed = random.seed(seed)

        def add(self, state, action, reward, next_state, done):
            """ Add a new experience to memory.

                INPUTS:
                ------------
                    state - (numpy array) with shape (33,) state vector for the actual agent
                    action - (numpy array) with shape (4,) actual action values for the agent
                    reward - (float) actual reward
                    next_state - (numpy array) with shape (33,) next state vector for the actual agent
                    done - (bool) if True epsiode is finished for the agent

                OUTPUTS:
                ------------
                    No direct
            """

            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

        def sample(self):
            """ Randomly sample a batch of experiences from memory.

                INPUTS:
                ------------
                    None

                OUTPUTS:
                ------------
                    states - (torch tensor)
                    actions - (torch tensor)
                    rewards - (torch tensor)
                    next_states - (torch tensor)
                    dones - (torch tensor)
            """

            experiences = random.sample(self.memory, k=self.batch_size)

            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

            return (states, actions, rewards, next_states, dones)

        def __len__(self):
            """ Return the current size of internal memory.

                INPUTS:
                ------------
                    None

                OUTPUTS:
                ------------
                    current_size - (int) length of self.memory
            """

            current_size = len(self.memory)
            return current_size
    ```


## Implementation - model.py <a name="impl_model"></a>
- Open Python file ```model.py```
- The code basis for **model.py** has been taken from the file **model.py** in the Udacity repo [ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)
    ### Import important libraries
    ```
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    ```
    ### Create a Pytorch Actor Policy model via a deep neural network 
    - **init** function: Initialize parameters and build model
    - **reset_parameters** function: Reset the parameters of the network (random uniform distribution)
    - **forward** function: 
        - create a forward pass, i.e. build a network that maps **state -> action values**
        - use tanh as an output activation function to limit action values between -1 and 1
    - **Architecture**:
        - Three fully connected layers
            - 1st hidden layer: fully connected, 33 input units, 256 output units, rectified via ReLU
            - 2nd hidden layer: fully connected, 256 input units, 128 output units, rectified via ReLU
            - 3rd hidden layer: fully connected, 128 input units, 4 output units, limited to the interval [-1,1] via tanh
    ```
    class Actor(nn.Module):
        """ Actor (Policy) Model
        """

        def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
            """ Initialize parameters and build model

                INPUTS:
                ------------
                    state_size - (int) Dimension of each state
                    action_size - (int) Dimension of each action
                    seed - (int) Random seed
                    fc1_units - (int) Number of nodes in first hidden layer
                    fc2_units - (int) Number of nodes in second hidden layer

                OUTPUTS:
                ------------
                    No direct
            """
            super(Actor, self).__init__()
            self.seed = torch.manual_seed(seed)
            self.fc1 = nn.Linear(state_size, fc1_units)
            #self.bn1 = nn.BatchNorm1d(fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            #self.bn2 = nn.BatchNorm1d(fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)
            self.reset_parameters()

        def reset_parameters(self):
            """ Reset the parameters of the network (random uniform distribution)

                INPUTS:
                ------------
                    None

                OUTPUTS:
                ------------
                    No direct
            """

            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        def forward(self, state):
            """ Build an actor (policy) network that maps states -> actions.

                INPUTS:
                ------------
                    state - (torch tensor) --> tensor([[33x floats], x size of minibatch])

                OUTPUTS:
                ------------
                    actions - (torch tensor) --> tensor([[4x floats], x size of minibatch])
            """

            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            actions = torch.tanh(self.fc3(x))
            return actions
    ```
    ### Create a Pytorch Critic (Value) model as a deep QNetwork
    - **init** function: Initialize parameters and build model
    - **reset_parameters** function: Reset the parameters of the network (random uniform distribution)
    - **forward** function: 
        - create a forward pass, i.e. build a network that maps **state -> Q-values**
        - use BatchNormalization (BatchNorm1d) to enhance covergence (faster learning) after the first linear layer
        - concat states with action values to create state-action pairs 
    - **Architecture**:
        - Three fully connected layers
            - 1st hidden layer: fully connected, 33 input units, 256 output units, batchnormalized, rectified via ReLU
            - Concat states with actions
            - 2nd hidden layer: fully connected, 256 + 4 input units, 128 output units, rectified via ReLU
            - 3rd hidden layer: fully connected, 128 input units, 1 output unit
    ```
    class Critic(nn.Module):
        """ Critic (Value) Model
        """

        def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
            """ Initialize parameters and build model

                INPUTS:
                ------------
                    state_size (int): Dimension of each state
                    action_size (int): Dimension of each action
                    seed (int): Random seed
                    fcs1_units (int): Number of nodes in the first hidden layer
                    fc2_units (int): Number of nodes in the second hidden layer

                OUTPUTS:
                ------------
                    No direct
            """

            super(Critic, self).__init__()
            self.seed = torch.manual_seed(seed)

            self.fcs1 = nn.Linear(state_size, fcs1_units)
            self.bn1 = nn.BatchNorm1d(fcs1_units)
            self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
            self.fc3 = nn.Linear(fc2_units, 1)
            self.reset_parameters()


        def reset_parameters(self):
            """ Reset the parameters of the network (random uniform distribution)

                INPUTS:
                ------------
                    None

                OUTPUTS:
                ------------
                    No direct
            """

            self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)


        def forward(self, state, action):
            """ Build a critic (value) network that maps (state, action) pairs -> Q-values.

                INPUTS:
                ------------
                    state - (torch tensor) --> tensor([[33x floats], x size of minibatch])
                    action - (torch tensor) --> tensor([[4x floats], x size of minibatch])

                OUTPUTS:
                ------------
                    Q_values - (torch tensor) --> tensor([[1x float], x size of minibatch])
            """

            xs = F.relu(self.bn1(self.fcs1(state)))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))
            Q_values = self.fc3(x)
            return Q_values
    ```
    ### Some remarks:
    - It has been found that a input configuration of 256 and 128 hidden nodes for the first and second hidden layer, both for actor and critic gave the best result with regard to training performance.
    - A low number of hidden units (in the range of 64 and lower) let to instabilities for later episodes (even with the introduction of target clipping).
    - Batch Normalization seems to further increase training stability. 


## Implementation - Continuous_Control_Trained_Agent.ipynb <a name="impl_notebook_trained_agent"></a> 
- Open Jupyter Notebook ```Continuous_Control_Trained_Agent.ipynb```
    ### Import important libraries
    - modul ***unityagents*** provides the Unity Environment. This modul is part and installed via requirements.txt. Check the README.md file for detailed setup instructions.
    - modul **dqn_agent** is the own implementation of an DQN agent. Check the description of **dqn_agent.py** for further details. 
    ```
    import gym
    import random
    import torch
    import numpy as np
    import time
    from collections import deque
    import matplotlib.pyplot as plt
    %matplotlib inline

    from unityagents import UnityEnvironment
    from ddpg_agent import Agent
    ```
    ### Instantiate the Environment
    ```
    # Load the Unity environment
    env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe', no_graphics=False, worker_id=1)
    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print()
    print('The state for the first agent looks like: \n', states[0])
    ```
    ### Watch a smart agent in action
    ```
    # test the trained agent
    agent = Agent(state_size=state_size, 
                action_size=action_size, 
                num_agents=num_agents, 
                random_seed=42, 
                checkpoint_actor='checkpoint_actor.pth',
                checkpoint_critic='checkpoint_critic.pth')


    for episode in range(4):
        env_info = env.reset(train_mode=False)[brain_name]        
        states = env_info.vector_observations       
        score = np.zeros(num_agents)               
        
        while True:
            actions = agent.act(states, add_noise=False)                    
            
            env_info = env.step(actions)[brain_name]        
            next_states = env_info.vector_observations     
            rewards = env_info.rewards       
            dones = env_info.local_done
            score += rewards
            states = next_states

            if np.any(dones):                              
                break

        print('Episode: \t{} \tScore: \t{:.2f}'.format(episode, np.mean(score)))
    ```
    - Pass the best trained model weights (checkpoint_actor.pth, checkpoint_critic.pth) for actor and critic to the Agent class instance
    - These weights will be loaded and used for testing.
    - The gif below visualizes a trained result.

        ![image1]

## Ideas for future work <a name="ideas_future"></a> 
Several parts of the actual implementation could be improved in the future:
- At the moment there is no asynchronous ([A3C](https://arxiv.org/pdf/1602.01783.pdf)) approach implemented, i.e. we do not use agent parallelization via multi-core CPU threading at the moment. Such approaches could speed up learning very efficiently. In addition, samples will be decorrelated because agents will likely experience different states at any given time. This means, we could remove the ReplayBuffer by using this approach.
- Further, more sophisticated hyperparameter tuning could speed up and stabilize the learning process.
- n-step bootstrapping instead of TD estimates for the critic network could result in faster convergence with less experience required and a reduction of the TD estimate bias.
- The implementation of Generalized Advantage Estimation ([GAE](https://arxiv.org/pdf/1506.02438.pdf)), i.e. a mixture implementation of all n-step bootstrapping estimates at once via the calculation of the exponantially **n**-decaying (for **n**-step bootstrapping) lambda return could further speed up training because multiple value functions spread around on every time step.