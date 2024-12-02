import random

from abc import ABC, abstractmethod
# from habitat_sim.simulator.Sensor import get_observation

class BaseModel(ABC):
    @abstractmethod
    def action(self, observation, action_list):
        """
        Choose an action from the action list based on the current state.
        
        Args:
            observation: The current state of the environment, include RGB, Depth, semantics
            action_list: List of possible actions to choose from
            
        Returns:
            action: Selected action from action_list
        """
        pass
    
class turnLeftModel(BaseModel):
    def action(self, observation, action_list):
        """
        Always returns 'turn_left' regardless of state or action_list.
        
        Args:
            state: The current state of the environment (ignored)
            
        Returns:
            str: Always returns 'turn_left'
        """
        print(f"test state: {observation} and action_list {action_list}")
        return 'turn_left'
    
class RandomActionModel(BaseModel):
    def action(self, observation, action_list):
        """
        Returns a random action from the action_list.
        
        Args:
            observation: The current state of the environment
            action_list: List of possible actions to choose from
            
        Returns:
            str: Randomly selected action from action_list
        """
        print(f"Current observation: {observation} and action_list {action_list}")
        return random.choice(action_list)