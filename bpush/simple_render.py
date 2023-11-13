import numpy as np
from PIL import Image

PIXEL_SCALE = 3
IMG_SCALE = 12

BASE_COLOUR = (0,0,255)
CHEQUER_V = 230

GOAL_COLOR = (  0,  0, 48)
BOULDER_COLOR = (155,  220, 154)
AGENT_COLOR = ( 24, 255, 255)


def _pixel_to_slice(pixel):
    return slice(pixel*PIXEL_SCALE, (pixel+1)*PIXEL_SCALE)


def _color_to_arr(color):
    return np.expand_dims(np.array(color, dtype=np.uint8), (1,2))


def render(env):
    """Renders the environment."""
    base_pixel = np.tile(
        _color_to_arr(BASE_COLOUR),
        (PIXEL_SCALE, PIXEL_SCALE)
        )
    grid_size = env.grid_size
    img = np.tile(base_pixel, grid_size)
    # chequer
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            if (x-y)%2 == 0:
                r = _pixel_to_slice(y)
                c = _pixel_to_slice(x)
                img[2,r,c] = CHEQUER_V

    # goals
    for (x,y) in zip(*np.nonzero(env.grid[2])):
        r = _pixel_to_slice(y)
        c = _pixel_to_slice(x)
        img[:,r,c] = _color_to_arr(GOAL_COLOR)

    # boulder
    for (x,y) in zip(*np.nonzero(env.grid[1])):
        r = _pixel_to_slice(y)
        c = _pixel_to_slice(x)
        img[:,r,c] = _color_to_arr(BOULDER_COLOR)

    # agents
    for (x,y) in zip(*np.nonzero(env.grid[0])):
        r = _pixel_to_slice(y)
        c = _pixel_to_slice(x)
        img[:,r,c] = _color_to_arr(AGENT_COLOR)

    rgb_img = Image.fromarray(
        img.transpose((2,1,0)), mode="HSV"
    ).convert(
        "RGB"
    ).resize(
        (IMG_SCALE*img.shape[2], IMG_SCALE*img.shape[1]),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(rgb_img)
