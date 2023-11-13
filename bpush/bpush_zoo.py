import logging

from collections import defaultdict, OrderedDict
from copy import copy
import functools
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec

from bpush.utils import MultiAgentActionSpace, MultiAgentObservationSpace

import random
from enum import IntEnum
import numpy as np

from typing import List, Tuple, Optional, Dict

from .environment import _VectorWriter, Direction, Entity, Agent, Boulder


_AXIS_Z = 0
_AXIS_Y = 1
_AXIS_X = 2

_LAYER_AGENTS = 0
_LAYER_BOULDER = 1
_LAYER_GOAL = 2

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def parallel_env(**kwargs):
    env = BoulderPush(**kwargs)
    return env

def raw_env(**kwargs):
    env = parallel_env(**kwargs)
    env = parallel_to_aec(env)
    return env

class BoulderPush(ParallelEnv):

    metadata = {
        "name": "bpush",
        "render.modes": ["human", "rgb_array"],
        "render_fps": 4,
        }

    def __init__(
        self,
        height: int,
        width: int,
        n_agents: int,
        sensor_range: int,
        penalty: float = 0.0,
        incentive: float = 0.1,
        push_direction: Direction = None,
        max_steps: Optional[int] = None,
        render_mode: str = "rgb_array",
        render_style: str = "simple",
    ):
        """The boulder-push environment

        :param height: The height (max y-coordinate) of the grid-world
        :type height: int
        :param width: The width (max x-coordinate) of the grid-world
        :type width: int
        :param n_agents: The number of agents (and also the size of the boulder)
        :type n_agents: int
        :param sensor_range: The range of perception of the agents
        :type sensor_range: int
        :param push_direction: Direction to push boulder in (or None for any direction)
        :type push_direction: Direction
        :param max_steps: Maximum number of timesteps
        :type max_steps: Optional[int]
        """
        self.grid_size = (width, height)

        self.push_penalty = penalty
        self.push_reward = incentive
        self.push_direction = push_direction

        self.n_agents = n_agents
        self.sensor_range = sensor_range
        self.reward_range = (0, 1)

        self._cur_steps = 0
        self.max_steps = max_steps

        self.grid = np.zeros((3, *self.grid_size), dtype=np.int32)

        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents_obj = {}

        self.goals = None
        self.boulder = None

        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2
        self._obs_length = len(Direction) + 2 * self._obs_sensor_locations

        self.renderer = None
        self.render_mode = render_mode
        self.render_style = render_style

    def _make_obs(self, agent_id):
        agent = self.agents_obj[agent_id]
        y_scale, x_scale = self.grid_size[0] - 1, self.grid_size[1] - 1

        min_x = agent.x - self.sensor_range
        max_x = agent.x + self.sensor_range + 1

        min_y = agent.y - self.sensor_range
        max_y = agent.y + self.sensor_range + 1
        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[1])
            or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_BOULDER], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_BOULDER]

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        boulder = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)

        obs = _VectorWriter(self.observation_space(agent_id).shape[0])

        obs.write(np.eye(4)[int(self.boulder.orientation)])
        obs.write(agents)
        obs.write(boulder)

        return obs.vector

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        self.seed(seed)
        Boulder.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9
        self.grid[:] = 0

        # Spawn a boulder..
        if self.push_direction is None:
            # First choose a random orientation (pushing boulder north, south, east, or west)
            push_towards = Direction(self.np_random.integers(0, 3))
        else:
            # Take set direction
            push_towards = self.push_direction

        if push_towards == Direction.SOUTH or push_towards == Direction.NORTH:
            x = self.np_random.integers(0, self.grid_size[1] - self.max_num_agents)
            y = self.np_random.integers(1, self.grid_size[0] - 2)
            self.grid[_LAYER_BOULDER, y, x : x + self.max_num_agents] = 1

        else:
            x = self.np_random.integers(1, self.grid_size[1] - 2)
            y = self.np_random.integers(0, self.grid_size[0] - self.max_num_agents)
            self.grid[_LAYER_BOULDER, y : y + self.max_num_agents, x] = 1

        # set goal
        if push_towards == Direction.SOUTH:
            self.grid[_LAYER_GOAL, self.grid_size[0] - 1, x : x + self.max_num_agents] = -1
        elif push_towards == Direction.NORTH:
            self.grid[_LAYER_GOAL, 0, x : x + self.max_num_agents] = -1
        elif push_towards == Direction.EAST:
            self.grid[_LAYER_GOAL, y : y + self.max_num_agents, self.grid_size[0] - 1] = -1
        elif push_towards == Direction.WEST:
            self.grid[_LAYER_GOAL, y : y + self.max_num_agents, 0] = -1

        # spawn the boulder
        self.boulder = Boulder(x, y, self.max_num_agents, push_towards)

        # spawn the agents
        self.agents = copy(self.possible_agents)
        self.agents_obj = {}
        for agent_id in self.agents:
            while True:
                x, y = self.np_random.integers(0, self.grid_size[1] - 1), self.np_random.integers(
                    0, self.grid_size[0] - 1
                )
                if (
                    self.grid[_LAYER_AGENTS, y, x] == 0
                    and self.grid[_LAYER_BOULDER, y, x] == 0
                ):
                    self.agents_obj[agent_id] = Agent(x, y)
                    self.grid[_LAYER_AGENTS, y, x] = 1
                    break

        obs = {agent: self._make_obs(agent) for agent in self.agents}
        infos = {agent_id: {} for agent_id in self.agents}
        return obs, infos

    def _draw_grid(self):
        self.grid[_LAYER_AGENTS, :] = 0
        self.grid[_LAYER_BOULDER, :] = 0

        if self.boulder.orientation in (Direction.SOUTH, Direction.NORTH):
            self.grid[
                _LAYER_BOULDER,
                self.boulder.y,
                self.boulder.x : self.boulder.x + self.num_agents,
            ] = 1
        else:
            self.grid[
                _LAYER_BOULDER,
                self.boulder.y : self.boulder.y + self.num_agents,
                self.boulder.x,
            ] = 1
        for agent in self.agents_obj.values():
            self.grid[_LAYER_AGENTS, agent.y, agent.x] = 1

    def step(self, actions):
        assert len(actions) == len(self.agents_obj)

        done = False
        terminated = False
        reward = np.zeros(self.num_agents, np.float32)
        # first check if the agents manage to push the boulder
        if (
            self.boulder.orientation == Direction.NORTH
            and self.grid[
                _LAYER_AGENTS,
                self.boulder.y + 1,
                self.boulder.x : self.boulder.x + self.num_agents,
            ].sum()
            == self.num_agents
            and all([a == Direction.NORTH for a in actions.values()])
        ):
            # pushing boulder north
            self.boulder.y -= 1
            for agent in self.agents_obj.values():
                agent.y -= 1
            self._draw_grid()
            reward += self.push_reward
            terminated = not self.grid[_LAYER_BOULDER:].sum(axis=0).any()

        elif (
            self.boulder.orientation == Direction.SOUTH
            and self.grid[
                _LAYER_AGENTS,
                self.boulder.y - 1,
                self.boulder.x : self.boulder.x + self.num_agents,
            ].sum()
            == self.num_agents
            and all([a == Direction.SOUTH for a in actions.values()])
        ):
            # pushing boulder south
            self.boulder.y += 1
            for agent in self.agents_obj.values():
                agent.y += 1
            self._draw_grid()
            reward += self.push_reward
            terminated = not self.grid[_LAYER_BOULDER:].sum(axis=0).any()

        elif (
            self.boulder.orientation == Direction.EAST
            and self.grid[
                _LAYER_AGENTS,
                self.boulder.y : self.boulder.y + self.num_agents,
                self.boulder.x - 1,
            ].sum()
            == self.num_agents
            and all([a == Direction.EAST for a in actions.values()])
        ):
            # pushing boulder east
            self.boulder.x += 1
            for agent in self.agents_obj.values():
                agent.x += 1
            self._draw_grid()
            reward += self.push_reward
            terminated = not self.grid[_LAYER_BOULDER:].sum(axis=0).any()

        elif (
            self.boulder.orientation == Direction.WEST
            and self.grid[
                _LAYER_AGENTS,
                self.boulder.y : self.boulder.y + self.num_agents,
                self.boulder.x + 1,
            ].sum()
            == self.num_agents
            and all([a == Direction.WEST for a in actions.values()])
        ):
            # pushing boulder west
            self.boulder.x -= 1
            for agent in self.agents_obj.values():
                agent.x -= 1
            self._draw_grid()
            reward += self.push_reward
            terminated = not self.grid[_LAYER_BOULDER:].sum(axis=0).any()

        else:
            # just move agents around
            # we will use classical MARL approach for resolving collisions:
            # in a random order, commit each agents move before moving to the next.
            # later agents (in the shuffled order) will be in a disadvantage
            for idx in sorted(range(self.num_agents), key=lambda _: self.np_random.random()):
                action = actions[self.agents[idx]]
                agent = self.agents_obj[self.agents[idx]]

                self.grid[_LAYER_AGENTS, agent.y, agent.x] = 0
                if (
                    action == Direction.NORTH
                    and agent.y > 0
                    and self.grid[_LAYER_AGENTS, agent.y - 1, agent.x]
                    == 0
                ):
                    if self.grid[_LAYER_BOULDER, agent.y - 1, agent.x] == 0:
                        agent.y -= 1
                    else:
                        reward[idx] -= self.push_penalty
                elif (
                    action == Direction.SOUTH
                    and agent.y < self.grid_size[0] - 1
                    and self.grid[_LAYER_AGENTS, agent.y + 1, agent.x] == 0
                ):
                    if self.grid[_LAYER_BOULDER, agent.y + 1, agent.x] == 0:
                        agent.y += 1
                    else:
                        reward[idx] -= self.push_penalty
                elif (
                    action == Direction.WEST
                    and agent.x > 0
                    and self.grid[_LAYER_AGENTS, agent.y, agent.x - 1] == 0
                ):
                    if self.grid[_LAYER_BOULDER, agent.y, agent.x - 1] == 0:
                        agent.x -= 1
                    else:
                        reward[idx] -= self.push_penalty

                elif (
                    action == Direction.EAST
                    and agent.x < self.grid_size[0] - 1
                    and self.grid[_LAYER_AGENTS, agent.y, agent.x + 1] == 0
                ):
                    if self.grid[_LAYER_BOULDER, agent.y, agent.x + 1] == 0:
                        agent.x += 1
                    else:
                        reward[idx] -= self.push_penalty
                
                self._draw_grid()

        self._cur_steps +=1
        reward += 1.0 if done else 0.0

        observations = {agent_id: self._make_obs(agent_id)
                        for agent_id in self.agents}
        rewards = dict(zip(self.agents, reward))
        truncated = (
            bool(self.max_steps)
            and self._cur_steps >= self.max_steps
        )
        terminations = {agent_id: terminated for agent_id in self.agents}
        truncations = {agent_id: truncated for agent_id in self.agents}
        infos = {agent_id: {} for agent_id in self.agents}
        return observations, rewards, terminations, truncations, infos

    def _init_render(self):
        if self.render_style == "full":
            from .rendering import Viewer
            self.renderer = Viewer(self.grid_size)
        elif self.render_style == "simple":
            from .simple_render import render
            self.renderer = render

    def render(self):
        if not self.renderer:
            self._init_render()
        if self.render_style == "full":
            return self.renderer.render(self, return_rgb_array=True)
        elif self.render_style == "simple":
            return self.renderer(self)




    def close(self):
        if self.render_style == "full":
            if self.renderer:
                self.renderer.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self._obs_length,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(Direction))
    

    @property
    def observation_spaces(self):
        return {agent: self.observation_space(agent)
                for agent in self.possible_agents}

    @property
    def action_spaces(self):
        return {agent: self.action_space(agent)
                for agent in self.possible_agents}


if __name__ == "__main__":
    env = BoulderPush(8, 8, 3, 3)
    env.reset()
    import time
    from tqdm import tqdm

    time.sleep(2)
    # env.render()
    # env.step(18 * [Action.LOAD] + 2 * [Action.NOOP])

    for _ in tqdm(range(1000000)):
        # time.sleep(2)
        # env.render()
        actions = env.action_space.sample()
        env.step(actions)
