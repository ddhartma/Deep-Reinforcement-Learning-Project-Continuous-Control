[image1]: assets/scoring_result.png "image1"

# Deep Reinforcement Learning Project - Unitiy-Banana-DQN - Implementation Report

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
    - **Number of agents (states.shape[0]):** 1
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
    3.78988951e-01]]
    states shape (1, 33)
    type <class 'numpy.ndarray'>

    actions  [[-1.         -1.         -0.61081709 -0.06546604]]
    actions shape (1, 4)
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
    3.78988951e-01]]
    next_states shape (1, 33)
    type <class 'numpy.ndarray'>

    rewards
    [0.0]
    type <class 'list'>

    dones
    [True]
    type <class 'list'>

    scores
    [0.13]
    scores shape (1,)
    type <class 'numpy.ndarray'>
    Total score (averaged over agents) this episode: 0.12999999709427357
    ```
    ### Instantiate the Agent
    ```
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

    RESULTS:
    ------------

    Training on  cpu

    self.state_size 33

    self.action_size 4

    --
    self.actor_local Actor(
    (fc1): Linear(in_features=33, out_features=400, bias=True)
    (batch_norm): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc2): Linear(in_features=400, out_features=300, bias=True)
    (fc3): Linear(in_features=300, out_features=4, bias=True)
    )

    self.actor_target Actor(
    (fc1): Linear(in_features=33, out_features=400, bias=True)
    (batch_norm): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc2): Linear(in_features=400, out_features=300, bias=True)
    (fc3): Linear(in_features=300, out_features=4, bias=True)
    )

    self.actor_optimizer Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        eps: 1e-08
        lr: 0.0001
        weight_decay: 0
    )

    self.critic_local Critic(
    (fcs1): Linear(in_features=33, out_features=400, bias=True)
    (batch_norm): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc2): Linear(in_features=404, out_features=300, bias=True)
    (fc3): Linear(in_features=300, out_features=1, bias=True)
    )

    self.critic_target Critic(
    (fcs1): Linear(in_features=33, out_features=400, bias=True)
    (batch_norm): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc2): Linear(in_features=404, out_features=300, bias=True)
    (fc3): Linear(in_features=300, out_features=1, bias=True)
    )

    self.critic_optimizer Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        eps: 1e-08
        lr: 0.001
        weight_decay: 0
    )

    self.noise <ddpg_agent.OUNoise object at 0x0000023490656780>

    self.epsilon 1.0
    ```
   
    ### Train an Agent
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
    def ddpg(n_episodes=100, max_t=1000, print_every=10):
        """ Deep Deterministic Policy Gradient
        
            INPUTS: 
            ------------
                n_episodes - (int) maximum number of training episodes
                max_t - (int) maximum number of timesteps per episode
                print_every (int) number episodes for printing average score 
                
            OUTPUTS:
            ------------
                scores - (list) list of score values for each episode
        """

        scores_deque = deque(maxlen=100)
        scores = []
        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]          
            states = env_info.vector_observations               
            agent.reset()
            score = np.zeros(num_agents)
            for t in range(max_t): 
                actions = agent.act(states)                        # return actions for current states and policy
                env_info = env.step(actions)[brain_name]           # send all actions to the environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                
                #for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(states[0], actions[0], rewards[0], next_states[0], dones[0])
                
                states = next_states
                score += rewards
                if np.any(dones):                                  # exit loop if episode finished
                    break
            
            scores_deque.append(score)
            scores.append(score)
            
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
            if i_episode % print_every == 0:
                print('\rEpisode {} Mean Score: {:.2f} Average Score: {:.2f}'.format(i_episode, np.mean(score), np.mean(scores_deque)))
            if np.mean(scores_deque) >= 30:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
                break
                
        return scores
    ```
    ### Plot the cumulated scores as a function of episodes
    ```
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    ```
    ![image1]
    ### Close the environment
    ```
    env.close()
    ```

## Implementation - ddpg_agent.py <a name="impl_agent"></a>
- Open Python file ```notebooks_python/ddpg_agent.py```
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
    LR_CRITIC = 1e-3        # learning rate of the critic
    WEIGHT_DECAY = 0        # L2 weight decay
    EPSILON = 1.0           # Epsilon value
    EPSILON_DECAY = 0.999999 # Epsilon Decay

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ```
    
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
                tensor([[33x floats for state], x self.batch_size for minibatch]),
                tensor([[4x int for action], x self.batch_size for minibatch]),
                tensor([[1x float for reward], x self.batch_size for minibatch]),
                tensor([[33x floats for next_state], x self.batch_size for minibatch]),
                tensor([[1x int for done], x self.batch_size for minibatch])
            )
            ```
        (- learn every UPDATE_EVERY time steps (by doing random sampling from Replaybuffer))
        - if ReplayBuffer is larger than self.batch_size sample from Replaybuffer 
    
    - **act** function:
        - returns a **clipped_action** for given state and current policy
        - **actor_local** is used to find the best action
        - convert **state** from numpy array to torch tensor
        - set **actor_local model** to evaluation mode (no optimizer step, no backpropagation)
        - get action values as Fixed Targets
        - set **actor_local model** back to train mode
        - optional: add Ornstein-Uhlenbeck noise 
        - initiate an epsilon greedy action selection
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
                    tensor([[1x float], x self.batch_size for minibatch])
                    ```
                - compute **Q_targets** for current states 
                    ```
                    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

                    RESULT structure for Q_targets:
                    tensor([[1x float], x self.batch_size for minibatch])
                    ```
                - get expected Q values **Q_expected** from **critic local model**
                    ```
                    Q_expected = self.critic_local(states, actions)

                    RESULT structure for Q_expected:
                    tensor([[1x float], x self.batch_size for minibatch])
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
        """Interacts with and learns from the environment."""

        def __init__(self, state_size, action_size, random_seed):
            """Initialize an Agent object.

                INPUTS:
                ------------
                    state_size - (int) dimension of each state
                    action_size - (int) dimension of each action
                    random_seed - (int) random seed

                OUTPUTS:
                ------------
                    No direct
            """
            print('Training on ', device)
            print()

            self.state_size = state_size
            self.action_size = action_size
            self.seed = random.seed(random_seed)

            # Actor Network (w/ Target Network)
            self.actor_local = Actor(state_size, action_size, random_seed).to(device)
            self.actor_target = Actor(state_size, action_size, random_seed).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

            # Critic Network (w/ Target Network)
            self.critic_local = Critic(state_size, action_size, random_seed).to(device)
            self.critic_target = Critic(state_size, action_size, random_seed).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

            # Noise process
            self.noise = OUNoise(action_size, random_seed)

            self.epsilon = EPSILON

            # Replay memory
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

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

        def step(self, state, action, reward, next_state, done):
            """ Save experience in replay memory, and use random sample from buffer to learn.

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

            # Save experience / reward
            self.memory.add(state, action, reward, next_state, done)

            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
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
                action += self.noise.sample()
                action *=  self.epsilon
            else:
                action *=  self.epsilon

            clipped_action = np.clip(action, -1, 1)

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

            # ----------------------- update epsilon / noise ----------------------- #
            self.epsilon *= EPSILON_DECAY
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
        - create a random **experiences** sample (of type list) from **self.memory** ReplayBuffer instance with size **self.batch_size**
        - structure of experience sample: list of named tuples **Experience** with length **self.batch_size**
            ```
            [
                **Experience**(
                            **state**=array([33 xfloat values]),
                            **action**=array([4 x float values]),
                            **reward**=1 x float,
                            **next_state=array([33 x floatvalues])
                            ),
                **Experience**(
                            **state**=array([33 xfloat values]),
                            **action**=array([4 x float values]),
                            **reward**=1 x float,
                            **next_state=array([33 x floatvalues])
                            ),
                ... 
            ]              
            ```
        - return **states**, **actions**, **rewards**, **next_states** and **dones** each as torch tensors
        - structure of 
            - **states**: torch tensor with shape (128,33)
            - **actions**: torch tensor with shape (128,4)
            - **rewards**: torch tensor with shape (128,1)
            - **next_states**: torch tensor with shape (128,33)
            - **dones**: torch tensor with shape (128,1)
            ```
            states: tensor([[ <float value,> x 33 ], x 128 ]) 
            actions: tensor([[ <float value,> x 4 ], x 128 ]) 
            rewards: tensor([[ <float value> ], x 128 ]) 
            next_states: tensor([[ <float value,> x 33 ], x 128 ]) 
            dones: tensor([[ <bool value> ], x 128 ]) 
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
                    state - ()
                    action - ()
                    reward - ()
                    next_state - ()
                    done - ()

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
    ### Import important libraries
    ```
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    ```
    ### Create a Pytorch Actor Policy model as a deep QNetwork
    - **init** function: Initialize parameters and build model
    - **reset_parameters** function: Reset the parameters of the network (random uniform distribution)
    - **forward** function: 
        - create a forward pass, i.e. build a network that maps **state -> action values**
        - use Batchnormalization to enhance covergence (faster learning)
        - use tanh as an output activation function to clip action values between -1 and 1
    - **Architecture**:
        - Three fully connected layers
            - 1st hidden layer: fully connected, 33 input units, fc1_units output units, rectified via ReLU
            - 2nd hidden layer: fully connected, fc1_units input units, fc2_units output units, rectified via ReLU
            - 3rd hidden layer: fully connected, fc2_units input units, 4 output units
    ```
    class Actor(nn.Module):
        """ Actor (Policy) Model."""

        def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
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
            self.batch_norm = nn.BatchNorm1d(fc1_units)       # Use batch normalization
            self.fc2 = nn.Linear(fc1_units, fc2_units)
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
                    state - (torch tensor) --> tensor([[33x floats for states], x size of minibatch])

                OUTPUTS:
                ------------
                    actions - (torch tensor) --> tensor([[4x floats], x size of minibatch])
            """
            x = F.relu(self.batch_norm(self.fc1(state)))
            x = F.relu(self.fc2(x))
            actions = F.tanh(self.fc3(x))
            return actions
    ```
    ### Create a Pytorch Critic (Value) model as a deep QNetwork
    - **init** function: Initialize parameters and build model
    - **reset_parameters** function: Reset the parameters of the network (random uniform distribution)
    - **forward** function: 
        - create a forward pass, i.e. build a network that maps **state -> Q-values**
        - use Batchnormalization to enhance covergence (faster learning)
        - concat states with action values to create state-action pairs 
    - **Architecture**:
        - Three fully connected layers
            - 1st hidden layer: fully connected, 33 input units, fcs1_units output units, rectified via ReLU
            - 2nd hidden layer: fully connected, fcs1_units input units, fc2_units output units, rectified via ReLU
            - 3rd hidden layer: fully connected, fc2_units input units, 1 output unit
    ```
    class Critic(nn.Module):
        """ Critic (Value) Model
        """

        def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
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
            self.batch_norm = nn.BatchNorm1d(fcs1_units)
            self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
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
            xs = F.relu(self.batch_norm(self.fcs1(state)))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))
            Q_values = self.fc3(x)
            return Q_values
    ```
    
## Implementation - Navigation_Trained_Agent.ipynb <a name="impl_notebook_trained_agent"></a> 
- Open Jupyter Notebook ```Navigation_Trained_Agent.ipynb```
    ### Import important libraries
    - modul ***unityagents*** provides the Unity Environment. This modul is part and installed via requirements.txt. Check the README.md file for detailed setup instructions.
    - modul **dqn_agent** is the own implementation of an DQN agent. Check the description of **dqn_agent.py** for further details. 
    ```
    import random
    import torch
    import numpy as np
    from collections import deque
    import matplotlib.pyplot as plt
    %matplotlib inline

    from unityagents import UnityEnvironment
    from dqn_agent import Agent
    ```
    ### Instantiate the Environment
    ```
    # Load the Unity environment
    env = UnityEnvironment(file_name="Banana.app")

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Get an instance of an agent from the Agent class (see module dqn_agent)
    agent = Agent(state_size=37, action_size=4, seed=0)

    # Load the weights from the pytorch state_dict file checkpoint.pth
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]             # get the current state
    score = 0                                           # initialize the score
    ```
    ### Watch a smart agent in action
    ```
    while True:
        action = agent.act(state)                      # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
        
    print("Score: {}".format(score))
    ```

## Ideas for future work <a name="ideas_future"></a> 
- Implement Deep Q-Learning Improvements like:
    - [Double Q-Learning](https://arxiv.org/abs/1509.06461): Deep Q-Learning [tends to overestimate](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf) action values. In early stages, the Q-values are still evolving. This can result in an overestimation of Q-values, since the maximum values are chosen from noisy numbers. Solution: Select the best action using one set of weights w, but evaluate it using a different set of weights w'. It's basically like having two separate function approximators.

    - [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952): Deep Q-Learning samples experience transitions uniformly from a replay memory. Prioritized experienced replay is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability.

    - [Dueling DQN](https://arxiv.org/abs/1511.06581): Currently, in order to determine which states are (or are not) valuable, we have to estimate the corresponding action values for each action. However, by replacing the traditional Deep Q-Network (DQN) architecture with a dueling architecture, we can assess the value of each state, without having to learn the effect of each action. The core idea of dueling networks is to use two streams, one that estimates the state value function and one that estimates the advantage for each action.

    - [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298): A Rainbow DQN algorithm combines the upper three modificartions (Double Q-Learning, Prioritized Experience Replay, Dueling DQN) together with:
        - Learning from [multi-step bootstrap targets](https://arxiv.org/abs/1602.01783) 
        - [Distributional DQN](https://arxiv.org/abs/1707.06887)
        - [Noisy DQN](https://arxiv.org/abs/1706.10295)

- Further Readings for DQN optimizations:
    - [Speeding up DQN on PyTorch: how to solve Pong in 30 minutes](https://shmuma.medium.com/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55)
    - [Advanced DQNs: Playing Pac-man with Deep Reinforcement Learning by mapping pixel images to Q values](https://towardsdatascience.com/advanced-dqns-playing-pac-man-with-deep-reinforcement-learning-3ffbd99e0814)
    - Interesting GitHub repo based on [Prioritized Experience Replay](https://github.com/rlcode/per)
    - [Conquering OpenAI Retro Contest 2: Demystifying Rainbow Baseline](https://medium.com/intelligentunit/conquering-openai-retro-contest-2-demystifying-rainbow-baseline-9d8dd258e74b)
  