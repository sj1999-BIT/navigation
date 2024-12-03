import random

from abc import ABC, abstractmethod
from habitat_sim.nav import ShortestPath
# from habitat_sim.simulator.Sensor import get_observation


"""
define models to be able to react to surroundings and determine the next action to take.
"""

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
    
    
class ShortestPathModel(BaseModel):
    
    def __init__(self, start_position, goal_position, simulator, agent_id):
        self._sim = simulator
        
        # we compute the list of points on the mesh to reach the goal
        self._pathway_points = self.compute_shortest_path(
            start_position, goal_position
        )
        
        # we can get a list of 
        self._actionPath = []
        
        # generate an action path for the agent to move to each point
        self._point_index = 0
        
        
        self._actionIndex = 0
        
        
        self.greedy_follower = self._sim.make_greedy_follower(agent_id=agent_id)
    
    def compute_shortest_path(self, start_pos, end_pos):
        """
        Calculate the shortest path between two points using the simulator's pathfinder.

        Args:
            start_pos: The starting position coordinates
            end_pos: The target end position coordinates

        Notes:
            - Updates the internal _shortest_path object with the requested start and end positions
            - Uses the simulator's pathfinder to compute the optimal path
            - Prints the geodesic distance (shortest path length) after computation

        The geodesic distance represents the length of the shortest possible path between
        the points, taking into account navigable areas and obstacles in the environment.
        """
        shortest_path = ShortestPath()
        shortest_path.requested_start = start_pos
        shortest_path.requested_end = end_pos
        self._sim.pathfinder.find_path(shortest_path)
        print("shortest_path.geodesic_distance", shortest_path.geodesic_distance)
        
        # return a list of points for the agent to follow in the env to reach goal
        return shortest_path.points
    
    def action(self, observation, action_list):
        """
        Utilise the shortestPath fucntion from habitat to navigate
        
        Args:
            observation: The current state of the environment
            action_list: List of possible actions to choose from
            
        Returns:
            str: Randomly selected action from action_list
        """
        
        # have not yet reach next point
        if self._actionIndex < len(self._actionPath):
            self._actionIndex += 1
            # return the next action
            return self._actionPath[self._actionIndex]
        
        # if we reach we check if we have the next navigation point need to reach
        
        if self._point_index < len(self._pathway_points):
            next_point = self._pathway_points[self._point_index]
            self._point_index += 1
            
            # reset to zero
            self._actionIndex = 0
            
            # locate next action path
            self._action_path = self.greedy_follower.find_path(next_point)
            
            print("len(action_path)", len(self._action_path))
            print(f"test current action path {self._action_path}")
            
        if len(self._actionPath) == 0:
            print(f"no more path left, just stop moving")
            return action_list[-1]
            
            
                
           
        # return the first action of the new action path
        return self._actionPath[self._actionIndex]
            