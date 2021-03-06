import sys
sys.path.append("..") # Adds higher directory to python modules path. (for MalmoPython.so)

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import os
import time
import json
import random
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt 
import numpy as np
from numpy.random import randint
from numpy.random import choice
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Hyperparameters
SIZE = 20
SIZE_LESS = 19
DIAMOND_DENSITY = .005
COAL_DENSITY = .015
OBS_SIZE = 5
MAX_EPISODE_STEPS = 100
MAX_GLOBAL_STEPS = 10000
REPLAY_BUFFER_SIZE = 10000
EPSILON_DECAY = .999
MIN_EPSILON = .1
BATCH_SIZE = 128
GAMMA = .9
TARGET_UPDATE = 100
LEARNING_RATE = 1e-4
START_TRAINING = 500
LEARN_FREQUENCY = 1
ACTION_DICT = {
    0: 'move 1',        # Move one block forward
    1: 'setYaw 0',      # Set yaw to 0
    2: 'setYaw 90',     # Set yaw to 90
    3: 'setYaw 180',    # Set yaw to 180
    4: 'setYaw 270',    # Set yaw to 270
    5: 'jump 0',        # Deactivate continuous jump up
    6: 'jump 1',        # Activate continuous jump up
    7: 'attack 1'       # Destroy block
}
FULL_AIR = 300
YPOS_START = 11
RESOURCES = ['coal', 'diamond']


# Q-Value Network
class QNetwork(nn.Module):

    def __init__(self, obs_size, action_size, hidden_size=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(np.prod(obs_size), hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, action_size)) 
        
    def forward(self, obs):
        """
        Estimate q-values given obs

        Args:
            obs (tensor): current obs, size (batch x obs_size)

        Returns:
            q-values (tensor): estimated q-values, size (batch x action_size)
        """
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)
        return self.net(obs_flat)


def GetMissionXML():

    grid = choice([0, 1, 2, 3, 4], size=(SIZE_LESS*2, SIZE_LESS*2), p=[.96, DIAMOND_DENSITY, COAL_DENSITY, 
        DIAMOND_DENSITY, COAL_DENSITY])
    resource_xml = ""

    for index, row in enumerate(grid):
        for col, item in enumerate(row):
            if item == 1:
                resource_xml += "<DrawBlock x='{}'  y='2' z='{}' type='diamond_ore' />".format(index-SIZE_LESS, col-SIZE_LESS)
            elif item == 2:
                resource_xml += "<DrawBlock x='{}'  y='2' z='{}' type='coal_ore' />".format(index-SIZE_LESS, col-SIZE_LESS)
            elif item == 3:
                resource_xml += "<DrawItem x='{}'  y='2' z='{}' type='diamond' />".format(index-SIZE_LESS, col-SIZE_LESS)
            elif item == 4:
                resource_xml += "<DrawItem x='{}'  y='2' z='{}' type='coal' />".format(index-SIZE_LESS, col-SIZE_LESS)
    
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                <About>
                    <Summary>Shallow Blue</Summary>
                </About>

                <ServerSection>
                    <ServerInitialConditions>

                        <Time>
                            <StartTime>6000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>

                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;7,2;1;"/>

                        <DrawingDecorator>
                            ''' + \
                            "<DrawCuboid x1='{}' x2='{}' y1='0' y2='20' z1='{}' z2='{}' type='sea_lantern'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='1' y2='20' z1='{}' z2='{}' type='air'/>".format(-SIZE_LESS, SIZE_LESS, -SIZE_LESS, SIZE_LESS) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='0' y2='2' z1='{}' z2='{}' type='sea_lantern'/>".format(-SIZE, SIZE, -SIZE, SIZE) + \
                            "<DrawCuboid x1='{}' x2='{}' y1='2' y2='10' z1='{}' z2='{}' type='water'/>".format(-SIZE_LESS, SIZE_LESS, -SIZE_LESS, SIZE_LESS) + \
                            resource_xml + \
                            '''
                            
                        </DrawingDecorator>

                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Survival">
                    <Name>ShallowBlue</Name>

                    <AgentStart>
                        <Placement x="0.5" y="11" z="0.5" pitch="45" yaw="0"/>
                        <Inventory>
                            <InventoryItem slot="0" type="diamond_pickaxe"/>
                        </Inventory>
                    </AgentStart>

                    <AgentHandlers>

                        <RewardForCollectingItem>
                            <Item reward="2" type="diamond"/>
                            <Item reward="0.5" type="coal"/>
                        </RewardForCollectingItem>

                        <RewardForMissionEnd rewardForDeath="-1"> 
                            <Reward reward="0" description=""></Reward>
                        </RewardForMissionEnd>
    
                        <DiscreteMovementCommands>
                            <ModifierList type="deny-list">
                                <command>jump</command>
                            </ModifierList>
                        </DiscreteMovementCommands>

                        <ContinuousMovementCommands>
                            <ModifierList type="allow-list">
                                <command>jump</command>
                            </ModifierList>
                        </ContinuousMovementCommands>

                        <AbsoluteMovementCommands/>

                        <ObservationFromFullStats/>
                        <ObservationFromRay/>
                        <ObservationFromHotBar/>

                        <ObservationFromNearbyEntities>
                            <Range name="entities" xrange="'''+str(SIZE_LESS)+'''" yrange="1" zrange="'''+str(SIZE_LESS)+'''" />
                        </ObservationFromNearbyEntities>

                        <ObservationFromGrid>
                            <Grid name="floorAll">
                                <min x="-'''+str(int(OBS_SIZE/2))+'''" y="-1" z="-'''+str(int(OBS_SIZE/2))+'''"/>
                                <max x="'''+str(int(OBS_SIZE/2))+'''" y="0" z="'''+str(int(OBS_SIZE/2))+'''"/>
                            </Grid>
                        </ObservationFromGrid>
                        
                        <AgentQuitFromReachingCommandQuota total="'''+str(MAX_EPISODE_STEPS)+'''" />

                    </AgentHandlers>
                </AgentSection>
            </Mission>'''


def get_action(obs, q_network, epsilon, allow_break_action, air_level):
    """
    Select action according to e-greedy policy

    Args:
        obs (np-array): current observation, size (obs_size)
        q_network (QNetwork): Q-Network
        epsilon (float): probability of choosing a random action

    Returns:
        action (int): chosen action [0, action_size)
    """

    # Prevent computation graph from being calculated
    with torch.no_grad():
        # Calculate Q-values fot each action
        obs_torch = torch.tensor(obs.copy(), dtype=torch.float).unsqueeze(0)
        action_values = q_network(obs_torch)

        # Remove attack/mine from possible actions if not facing a diamond or coal
        if not allow_break_action:
            action_values[0, 7] = -float('inf')  
        
        # Remove swim up from possible actions if air > 30% of full
        if air_level > (FULL_AIR * 0.3): 
            action_values[0, 6] = -float('inf')
        else:
            action_values[0, 5] = -float('inf')

        explore = random.random() < epsilon

        if explore:
            if air_level > (FULL_AIR * 0.5):
                if not allow_break_action:
                    action_idx = random.randint(0,5)                # removes action 6 (swim up) and 7 (attack)
                else:
                    action_idx = random.choice([0,1,2,3,4,5,7])     # removes action 6 (swim up)

            else:
                if not allow_break_action:
                    action_idx = random.choice([0,1,2,3,4,6])       # removes action 5 (cancel swim up) and 7 (attack)
                else:
                    action_idx = random.choice([0,1,2,3,4,6,7])     # removes action 5 (cancel swim up)
        else:
            # Select action with highest Q-value
            action_idx = torch.argmax(action_values).item()
    
    return action_idx


def init_malmo(agent_host):
    """
    Initialize new malmo mission.
    """
    my_mission = MalmoPython.MissionSpec(GetMissionXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)

    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_clients, my_mission_record, 0, "ShallowBlue" )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    return agent_host


# inspired by fellow team MineFarm-Farmer: https://yongfeitan.github.io/MineFarm-Farmer/status.html
def get_entities(entities, agentX, agentZ):
    resources = dict()
    for ele in entities:
        if ele['name'] in RESOURCES:
            X = ele['x']
            Y = ele['y']
            Z = ele['z']

            # if agent is center of 5x5 square, any entity can only be at most 2.5 units away
            if _dist(agentX, agentZ, X, Z) <= 2.5:
                id = ele['id']
                name = ele['name']
                resources[id] = {'name':name,'x':X,'y':Y,'z':Z}
    return resources

def _dist(x1,z1,x2,z2):
    return ((x1-x2)**2 + (z1-z2)**2) ** 0.5


def get_observation(world_state):
    """
    Use the agent observation API to get a 2 x 5 x 5 grid around the agent. 
    The agent is in the center square facing up.

    Args
        world_state: <object> current agent world state

    Returns
        observation: <np.array>
    """

    obs = np.zeros((2, OBS_SIZE, OBS_SIZE))
    air_level = 0
    has_resources = False

    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            # First we get the json from the observation API
            msg = world_state.observations[-1].text
            observations = json.loads(msg)

            # Get observation
            grid = observations['floorAll']
            air_level = observations['Air']
            has_resources = observations['Hotbar_1_size'] > 0 or observations['Hotbar_2_size'] > 0
            agentX = observations['XPos']
            agentZ = observations['ZPos']
            resources = get_entities(observations['entities'], agentX, agentZ)

            # calculates the row and col in the 5x5 obs array from the entity's (x,z)
            for entity in resources.values():
                x = entity['x']
                z = entity['z']

                col_dist = abs(agentX - x)
                if x < agentX:
                    col = 2 - col_dist  # 2 is center column of the grid
                else:
                    col = 2 + col_dist

                row_dist = abs(agentZ - z)
                if z > agentZ:
                    row = 2 - row_dist  # 2 is center row of the grid
                else:
                    row = 2 + row_dist

                # unsure if these observations should go in obs[0] or obs[1]
                obs[0, int(row), int(col)] = 1

            grid_binary = [1 if x == 'diamond_ore' or x == 'coal_ore' else 0 for x in grid]
            grid_obs = np.reshape(grid_binary, (2, OBS_SIZE, OBS_SIZE))

            # obs = 1 where theres a loose resource OR an ore block (multiply by 1 to convert from boolean to int)
            obs = np.logical_or(obs, grid_obs) * 1

            # Rotate observation with orientation of agent
            yaw = observations['Yaw']
            if yaw == 270:
                obs = np.rot90(obs, k=1, axes=(1, 2))
            elif yaw == 0:
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif yaw == 90:
                obs = np.rot90(obs, k=3, axes=(1, 2))
            
            break

    return obs, air_level, has_resources


def prepare_batch(replay_buffer):
    """
    Randomly sample batch from replay buffer and prepare tensors

    Args:
        replay_buffer (list): obs, action, next_obs, reward, done tuples

    Returns:
        obs (tensor): float tensor of size (BATCH_SIZE x obs_size
        action (tensor): long tensor of size (BATCH_SIZE)
        next_obs (tensor): float tensor of size (BATCH_SIZE x obs_size)
        reward (tensor): float tensor of size (BATCH_SIZE)
        done (tensor): float tensor of size (BATCH_SIZE)
    """
    batch_data = random.sample(replay_buffer, BATCH_SIZE)
    obs = torch.tensor([x[0] for x in batch_data], dtype=torch.float)
    action = torch.tensor([x[1] for x in batch_data], dtype=torch.long)
    next_obs = torch.tensor([x[2] for x in batch_data], dtype=torch.float)
    reward = torch.tensor([x[3] for x in batch_data], dtype=torch.float)
    done = torch.tensor([x[4] for x in batch_data], dtype=torch.float)
    
    return obs, action, next_obs, reward, done
  

def learn(batch, optim, q_network, target_network):
    """
    Update Q-Network according to DQN Loss function

    Args:
        batch (tuple): tuple of obs, action, next_obs, reward, and done tensors
        optim (Adam): Q-Network optimizer
        q_network (QNetwork): Q-Network
        target_network (QNetwork): Target Q-Network
    """
    obs, action, next_obs, reward, done = batch

    optim.zero_grad()
    values = q_network(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
    target = torch.max(target_network(next_obs), 1)[0]
    target = reward + GAMMA * target * (1 - done)
    loss = torch.mean((target - values) ** 2)
    loss.backward()
    optim.step()

    return loss.item()


def log_returns(steps, returns):
    """
    Log the current returns as a graph and text file

    Args:
        steps (list): list of global steps after each episode
        returns (list): list of total return of each episode
    """
    box = np.ones(10) / 10
    returns_smooth = np.convolve(returns, box, mode='same')
    plt.clf()
    plt.plot(steps, returns_smooth)
    plt.title('Shallow Blue')
    plt.ylabel('Return')
    plt.xlabel('Steps')
    plt.savefig('returns.png')

    with open('returns.txt', 'w') as f:
        for value in returns:
            f.write("{}\n".format(value)) 


def train(agent_host):
    """
    Main loop for the DQN learning algorithm

    Args:
        agent_host (MalmoPython.AgentHost)
    """
    # Init networks
    q_network = QNetwork((2, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network = QNetwork((2, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network.load_state_dict(q_network.state_dict())

    # Init optimizer
    optim = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    # Init replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Init vars
    global_step = 0
    num_episode = 0
    epsilon = 1
    start_time = time.time()
    returns = []
    steps = []

    # Begin main loop
    loop = tqdm(total=MAX_GLOBAL_STEPS, position=0, leave=False)
    while global_step < MAX_GLOBAL_STEPS:
        episode_step = 0
        episode_return = 0
        episode_loss = 0
        done = False
        air_level = 0
        has_resources = False

        # Setup Malmo
        agent_host = init_malmo(agent_host)
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:",error.text)
        obs, air_level, has_resources = get_observation(world_state)

        # Run episode
        while world_state.is_mission_running:
            # Get action
            allow_break_action = obs[1, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 1
            action_idx = get_action(obs, q_network, epsilon, allow_break_action, air_level)
            command = ACTION_DICT[action_idx]

            # Take step
            agent_host.sendCommand(command)

            # If your agent isn't registering reward you may need to increase this
            time.sleep(.25)

            # We have to manually calculate terminal state to give malmo time to register the end of the mission
            # If you see "commands connection is not open. Is the mission running?" you may need to increase this
            episode_step += 1
            if episode_step >= MAX_EPISODE_STEPS or \
                    (obs[0, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 1 and \
                    obs[1, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 0 and \
                    command == 'move 1'):
                done = True
                time.sleep(5)  

            # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            next_obs, air_level, temp_has_resources = get_observation(world_state)

            # if has resources is already true, dont set it again
            if not has_resources and temp_has_resources:
                has_resources = True 

            # Get reward
            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()
            episode_return += reward

            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs

            # Learn
            global_step += 1
            if global_step > START_TRAINING and global_step % LEARN_FREQUENCY == 0:
                batch = prepare_batch(replay_buffer)
                loss = learn(batch, optim, q_network, target_network)
                episode_loss += loss

                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY

                if global_step % TARGET_UPDATE == 0:
                    target_network.load_state_dict(q_network.state_dict())

        # if agent hasnt collected anything, reduce its reward
        # if not has_resources:
        #     episode_return -= 0.5

        print("Has resources:",has_resources, "Score:", episode_return)

        num_episode += 1
        returns.append(episode_return)
        steps.append(global_step)
        avg_return = sum(returns[-min(len(returns), 10):]) / min(len(returns), 10)
        loop.update(episode_step)
        loop.set_description('Episode: {} Steps: {} Time: {:.2f} Loss: {:.2f} Last Return: {:.2f} Avg Return: {:.2f}'.format(
            num_episode, global_step, (time.time() - start_time) / 60, episode_loss, episode_return, avg_return))

        if num_episode > 0 and num_episode % 10 == 0:
            log_returns(steps, returns)
            print()


if __name__ == '__main__':
    # Create default Malmo objects:
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    train(agent_host)