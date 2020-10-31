---
layout: default
title:  Proposal
---

## 2.2 Summary of the Project

We aim to create an agent that gathers underwater resources in the most efficient way possible. The agent will receive observations from its environment as input in the form of the types of blocks two squares in any direction. It will then take appropriate actions, such as heading to the surface of the water to replenish air, avoiding obstacles and hostile mobs, and collecting resources on the ocean floor. The environment will start out as a flat 80x80 block seabed with a depth of 20 blocks, but will vary to include more challenges, including underwater structures, partially covered surfaces (to make getting air more difficult), items that replenish hunger and air levels, and more. This agent could potentially be used by Minecraft players to automate the repetitive task of gathering aquatic materials such as sand, clay, and food items, including various fish.

## 2.3 AI/ML Algorithms

We anticipate using reinforcement learning, specifically deep Q-Learning, with rewards and punishments. We will start with a Neural Network with linear layers to implement the Q-Network as done in Assignment 2. 
 
## 2.4 Evaluation Plan

Quantitative metrics for the performance of our agent include the number of resources gathered, value (determined by a reward system assigning numerical values to each type of resource) of the resources, whether the player survives, and the player’s resulting health level. We combine all of this into an overall score. If the agent dies, the score will be -1 regardless of the agent’s previous actions. If the agent is still alive at the end of 60 seconds, the score will be some weighted formula of the other metrics listed above. For example, picking up a diamond will be worth more than picking up dirt. Additionally, if the agent returns with low health and takes a long time for that particular environment, its overall score will be low. As we iterate on our project, we may modify the reward function. 

We plan to start with 3 different items: iron, gold, and diamonds, that are worth 1, 2, and 3 points respectively. For example, with an 80x80x20 environment (similar to a swimming pool) that has 10 iron, 10 gold, and 10 diamonds, our agent has 60 seconds to gather items and return to the surface. An estimate of our agent’s optimal behavior is that the 10 blocks of diamond are quickly gathered and the agent returns with minimal health lost. This would give it a reward of 30 points.

As a sanity check, the agent should be able to pick up a single object and then come up for air. Our case for the upcoming status report would be that given two items, one more valuable than the other, the agent prioritizes gathering the more valuable one. The “moonshot” case for this agent would be gathering objects and returning to the surface from a complex underwater ruins, when it will be more difficult for the agent to locate resources while maintaining its air level. We plan to visualize our agent’s learning rate by plotting its average return over episodes, similar to what was done in Assignment2.

## 2.5 Appointment with the Instructor
10/21 at 3:00 - 3:15pm

## Weekly Team Meetings
Sundays at 1pm
