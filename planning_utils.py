from enum import Enum
from queue import PriorityQueue
import numpy as np


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        """
        cost 1 for each step to any direction
        """
        return self.value[2]

    @property
    def delta(self):
        """
        return the first two value
        """
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    # NOTE: 1 means obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions


def a_star(grid, h, start, goal):
    print("\na_star\nfrom {0} to {1}\n".format(start, goal))

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get() # NOTE: use priority queue to sort by the g(n) + h(n)
        current_node = item[1] # NOTE: get the coordinate
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost


# Question 2
def iterative_astar(grid, h, start, goal):
    print("\niterative_astar\nfrom {0} to {1}\n".format(start, goal))

    path = []
    path_cost = 0
    # NOTE: check start == goal
    if start == goal: return path, path_cost

    exceed_f_queue = PriorityQueue()
    # NOTE: f(n) = g(n) + h(n)
    # NOTE: put (f_cost, coordinate)

    leaf_node = []
    leaf_node.append((0, start))
    # NOTE: put (f_cost, coordinate), simply set 0 not f(n) here just for convenience because it's the start point

    branch = {}
    found = False

    thres = 0 # NOTE: f_cost(start)
    while True:
        if len(leaf_node) == 0 and exceed_f_queue.empty(): break

        if len(leaf_node) == 0:
            item = exceed_f_queue.get()
            thres = item[0]
            while not exceed_f_queue.empty(): 
                leaf_node.append(exceed_f_queue.get()) # NOTE: go over the points with new threshold later
            exceed_f_queue = PriorityQueue() # NOTE: clear queue
        else:
            item = leaf_node.pop()

        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break

        if current_node == start: current_cost = 0.0
        else: current_cost = branch[current_node][0]

        for action in valid_actions(grid, current_node):
            next_node = (current_node[0] + action.delta[0], current_node[1] + action.delta[1])
            branch_cost = current_cost + action.cost # NOTE: cost(start -> next_node)
            f_cost = branch_cost + h(next_node, goal)

            # NOTE: check if the node is visited previously,
            # NOTE: here we don't need to concern that branch_cost_old > branch_cost_new
            # NOTE: because branch_cost = current_cost + action.cost (ac is always 1)
            if branch.get(next_node) == None:
                branch[next_node] = (branch_cost, current_node, action) # NOTE: current_node is the parent of the next_node
            
                if f_cost > thres:
                    exceed_f_queue.put((f_cost, next_node))
                else: leaf_node.append((f_cost, next_node))
    

    if not found:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
        return path, path_cost

    n = goal
    path_cost = branch[n][0]
    path.append(goal)
    while branch[n][1] != start:
        path.append(branch[n][1])
        n = branch[n][1]
    path.append(branch[n][1])
    return path[::-1], path_cost


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


# Question 3
def heuristic_new(position, goal_position):
    return 0.5*(abs(position[0]-goal_position[0]) + abs(position[1]-goal_position[1]))


# Question 4
def a_star_traverse(grid, h, traverse_points:list, start, goal):
    """
    find a path first traversing points in the order of the given list before reaching the goal with A* algorithm.

    params:
        grid, h, start, goal: same as a_star function
        traverse_points: the points needed to traverse before getting to the goal
    returns:
        if path found, return path and path cost; if not, return empty list and 0.
    """

    print("\na_star_traverse\nfrom {0} to {1}\n".format(start, goal))

    path = []
    path_cost = 0

    traverse_points.append(goal)
    traverse_points = traverse_points[::-1]
    while len(traverse_points) != 0:
        if len(path) != 0:
            start = goal
        goal = traverse_points.pop()

        print("\nfrom {0} to {1}\n".format(start, goal))

        queue = PriorityQueue()
        queue.put((0, start))
        visited = set()
        visited.add(start)

        branch = {}
        found = False
        
        while not queue.empty():
            item = queue.get() # NOTE: use priority queue to sort by g(n) + h(n)
            current_node = item[1] # NOTE: get the coordinate
            if current_node == start:
                current_cost = 0.0
            else:              
                current_cost = branch[current_node][0]
                
            if current_node == goal:        
                print('Found a path.')
                found = True
                break
            else:
                for action in valid_actions(grid, current_node):
                    # get the tuple representation
                    da = action.delta
                    next_node = (current_node[0] + da[0], current_node[1] + da[1])
                    branch_cost = current_cost + action.cost
                    queue_cost = branch_cost + h(next_node, goal)
                    
                    if next_node not in visited:                
                        visited.add(next_node)               
                        branch[next_node] = (branch_cost, current_node, action)
                        queue.put((queue_cost, next_node))

        if not found:
            print('**********************')
            print('Failed to find a path!')
            print('**********************')
            path = []
            path_cost = 0
            break

        # retrace steps
        temp_path = []
        n = goal
        path_cost = branch[n][0]
        temp_path.append(goal)
        while branch[n][1] != start:
            temp_path.append(branch[n][1])
            n = branch[n][1]
        if len(path) == 0: temp_path.append(branch[n][1])
        path += temp_path[::-1]

    return path, path_cost
