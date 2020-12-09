# Rllib docs: https://docs.ray.io/en/latest/rllib.html

import sys
sys.path.append("..") # Adds higher directory to python modules path. (for MalmoPython.so)

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import time
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from numpy.random import choice


import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo


class ShallowBlue(gym.Env):


    def __init__(self, env_config):  
        # Static Parameters
        self.size = 20
        self.size_pool = 19
        self.diamond_density = .005
        self.coal_density = .015
        self.obs_size = 5
        self.full_air = 300
        self.ypos_start = 11
        self.resource_list = {'coal', 'diamond'}
        self.max_episode_steps = 100
        self.log_frequency = 10
        self.action_dict = {
            0: 'move 1',        # Move one block forward
            1: 'setYaw 0',      # Set yaw to 0
            2: 'setYaw 90',     # Set yaw to 90
            3: 'setYaw 180',    # Set yaw to 180
            4: 'setYaw 270',    # Set yaw to 270
            5: 'jump 0',        # Deactivate continuous jump up
            6: 'jump 1',        # Activate continuous jump up
            7: 'attack 1'       # Destroy block
        }

        # Rllib Parameters
        self.action_space = Discrete(len(self.action_dict))
        self.observation_space = Box(0, 1, shape=(np.prod([2, self.obs_size, self.obs_size]), ), dtype=np.int32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # ShallowBlue Parameters
        self.obs = None
        self.air_level = 300
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []

        # reminder to implement penalty for 0 resources picked up if needed
        self.has_resources = False 


    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0

        # Log
        if len(self.returns) > self.log_frequency and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs = self.get_observation(world_state)

        return self.obs.flatten()


    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        # Get Action
        command = self.action_dict[action]
        allow_break_action = self.obs[1, int(self.obs_size/2)-1, int(self.obs_size/2)] == 1

        if command == 'attack 1' and allow_break_action:
            self.agent_host.sendCommand(command)
            time.sleep(.1)
            self.episode_step += 1
        elif command == 'attack 1':
            pass
        elif command == 'jump 1' and self.air_level > (self.full_air * .3): 
            pass
        elif command == 'jump 0' and self.air_level < (self.full_air * .3):
            pass
        else:
            self.agent_host.sendCommand(command)
            time.sleep(.1)
            self.episode_step += 1
            

        # Get Done
        done = False
        if self.episode_step >= self.max_episode_steps or \
                (self.obs[0, int(self.obs_size/2)-1, int(self.obs_size/2)] == 1 and \
                self.obs[1, int(self.obs_size/2)-1, int(self.obs_size/2)] == 0 and \
                command == 'move 1'):
            done = True
            time.sleep(2)  

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state) 

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episode_return += reward

        return self.obs.flatten(), reward, done, dict()


    def get_mission_xml(self):
        grid = choice([0, 1, 2, 3, 4], size=(self.size_pool*2, self.size_pool*2), p=[.96, self.diamond_density, self.coal_density, 
            self.diamond_density, self.coal_density])
        resource_xml = ""

        for index, row in enumerate(grid):
            for col, item in enumerate(row):
                if item == 1:
                    resource_xml += "<DrawBlock x='{}'  y='2' z='{}' type='diamond_ore' />".format(index-self.size_pool, col-self.size_pool)
                if item == 2:
                    resource_xml += "<DrawBlock x='{}'  y='2' z='{}' type='coal_ore' />".format(index-self.size_pool, col-self.size_pool)
                if item == 3:
                    resource_xml += "<DrawItem x='{}'  y='2' z='{}' type='diamond' />".format(index-self.size_pool, col-self.size_pool)
                elif item == 4:
                    resource_xml += "<DrawItem x='{}'  y='2' z='{}' type='coal' />".format(index-self.size_pool, col-self.size_pool)


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
                                "<DrawCuboid x1='{}' x2='{}' y1='0' y2='20' z1='{}' z2='{}' type='sea_lantern'/>".format(-self.size, self.size, -self.size, self.size) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='1' y2='20' z1='{}' z2='{}' type='air'/>".format(-self.size_pool, self.size_pool, -self.size_pool, self.size_pool) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='0' y2='2' z1='{}' z2='{}' type='sea_lantern'/>".format(-self.size, self.size, -self.size, self.size) + \
                                "<DrawCuboid x1='{}' x2='{}' y1='2' y2='10' z1='{}' z2='{}' type='water'/>".format(-self.size_pool, self.size_pool, -self.size_pool, self.size_pool) + \
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
                                <Range name="entities" xrange="'''+str(self.size_pool)+'''" yrange="1" zrange="'''+str(self.size_pool)+'''" />
                            </ObservationFromNearbyEntities>

                            <ObservationFromGrid>
                                <Grid name="floorAll">
                                    <min x="-'''+str(int(self.obs_size/2))+'''" y="-1" z="-'''+str(int(self.obs_size/2))+'''"/>
                                    <max x="'''+str(int(self.obs_size/2))+'''" y="0" z="'''+str(int(self.obs_size/2))+'''"/>
                                </Grid>
                            </ObservationFromGrid>
                            
                            <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''" />

                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''


    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'ShallowBlue' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state


    # inspired by fellow team MineFarm-Farmer: https://yongfeitan.github.io/MineFarm-Farmer/status.html
    def get_entities(self, entities, agentX, agentZ):
        resources = dict()
        for ele in entities:
            if ele['name'] in self.resource_list:
                X = ele['x']
                Y = ele['y']
                Z = ele['z']

                # if agent is center of 5x5 square, any entity can only be at most 2.5 units away
                if self._dist(agentX, agentZ, X, Z) <= 2.5:
                    id = ele['id']
                    name = ele['name']
                    resources[id] = {'name':name,'x':X,'y':Y,'z':Z}
        return resources

    def _dist(self, x1,z1,x2,z2):
        return ((x1-x2)**2 + (z1-z2)**2) ** 0.5
    

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array>
        """
        obs = np.zeros((2, self.obs_size, self.obs_size))

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)

                # Get observation
                grid = observations['floorAll']
                self.air_level = observations['Air']
                agentX = observations['XPos']
                agentZ = observations['ZPos']
                resources = self.get_entities(observations['entities'], agentX, agentZ)

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
                grid_obs = np.reshape(grid_binary, (2, self.obs_size, self.obs_size))

                # obs = 1 where theres a loose resource OR an ore block
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

        return obs


    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns, box, mode='same')
        plt.clf()
        plt.plot(self.steps, returns_smooth)
        plt.title('Diamond Collector')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps, self.returns):
                f.write("{}\t{}\n".format(step, value)) 


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=ShallowBlue, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())