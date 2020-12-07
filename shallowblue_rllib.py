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
        self.diamond_density = .01
        self.coal_density = .03
        self.obs_size = 5
        self.full_air = 300
        self.ypos_start = 11
        self.max_episode_steps = 200
        self.log_frequency = 10
        # 4 kinds of actions: [move, turn, jump, attack]
  
        # Rllib Parameters  
        self.action_space = Box(-1, 1, shape=(4,), dtype=np.float32)
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
        self.allow_break_action = False
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []

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
        self.obs, self.allow_break_action = self.get_observation(world_state)

        return self.obs.flatten()

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: 4 tuple corresponding to continous value of each action

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """
        # allow_break_action = self.obs[1, int(self.obs_size/2)-1, int(self.obs_size/2)] == 1

        # Get Action
        if self.allow_break_action and action[3] > 0:
            self.agent_host.sendCommand('move 0')
            self.agent_host.sendCommand('turn 0')
            self.agent_host.sendCommand('jump 0')
            self.agent_host.sendCommand('attack 1')
            time.sleep(1.5)
        elif (self.air_level < 0.3 * self.full_air) and action[2] > 0:
            self.agent_host.sendCommand('move 0')
            self.agent_host.sendCommand('turn 0')
            self.agent_host.sendCommand('attack 0')
            self.agent_host.sendCommand('jump 1')
            time.sleep(0.2)
        else:
            self.agent_host.sendCommand('jump 0')
            self.agent_host.sendCommand('attack 0')
            self.agent_host.sendCommand(f'move {action[0]:30.1}')
            self.agent_host.sendCommand(f'turn {action[1]:30.1}')
            time.sleep(0.2)
        self.episode_step += 1

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs, self.allow_break_action = self.get_observation(world_state) 

        # Get Done
        done = not world_state.is_mission_running

        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episode_return += reward

        return self.obs.flatten(), reward, done, dict()

    def get_mission_xml(self):
        grid = choice([0, 1, 2], size=(self.size_pool*2, self.size_pool*2), p=[.96, self.diamond_density, self.coal_density])
        diamond_xml = ""
        coal_xml = ""

        for index, row in enumerate(grid):
            for col, item in enumerate(row):
                if item == 1:
                    diamond_xml += "<DrawBlock x='{}'  y='2' z='{}' type='diamond_ore' />".format(index-self.size_pool, col-self.size_pool)
                elif item == 2:
                    diamond_xml += "<DrawBlock x='{}'  y='2' z='{}' type='coal_ore' />".format(index-self.size_pool, col-self.size_pool)
        
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
                                diamond_xml + \
                                coal_xml + \
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
                        
                            <ContinuousMovementCommands/>

                            <RewardForCollectingItem>
                                <Item reward="2" type="diamond"/>
                                <Item reward="0.5" type="coal"/>
                            </RewardForCollectingItem>

                            <RewardForMissionEnd rewardForDeath="-1"> 
                                <Reward reward="0" description="Mission End"></Reward>
                            </RewardForMissionEnd>
                    
                            <ObservationFromFullStats/>
                            <ObservationFromRay/>
                            <ObservationFromHotBar/>

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
        allow_break_action = False

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
                grid_binary = [1 if x == 'diamond_ore' or x == 'coal_ore' else 0 for x in grid]
                obs = np.reshape(grid_binary, (2, self.obs_size, self.obs_size))

                # Rotate observation with orientation of agent
                yaw = observations['Yaw']
                if yaw == 270:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw == 0:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw == 90:
                    obs = np.rot90(obs, k=3, axes=(1, 2))

                allow_break_action = observations['LineOfSight']['type'] == 'diamond_ore' or \
                    observations['LineOfSight']['type'] == 'coal_ore'
                
                break

        return obs, allow_break_action

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
        plt.title('Shallow Blue')
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
