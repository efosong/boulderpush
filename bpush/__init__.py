import gym
from .bpush_zoo import parallel_env
from .environment import Direction


_sizes = {
    "tiny": (5, 5),
    "small": (8, 8),
    "medium": (12, 12),
    "large": (20, 20),
}

_directions = {
    "": None,
    "north": 0,
    "south": 1,
    "west": 2,
    "east": 3,
}

for direction in _directions.keys():
    for size in _sizes.keys():
        for n_agents in range(1, 5):
            gym.register(
                id="-".join(filter(None, ["bpush",
                                          size,
                                          direction,
                                          f"{n_agents}ag",
                                          "v0"
                                          ])),
                entry_point="bpush.environment:BoulderPush",
                kwargs={
                    "height": _sizes[size][0],
                    "width": _sizes[size][1],
                    "n_agents": n_agents,
                    "sensor_range": 4,
                    "push_direction": _directions[direction],
                },
            )
