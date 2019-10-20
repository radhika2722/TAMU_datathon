from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import *
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import random
import itertools as it
import math

image_map_source_filename = '1150.png' # The store map file location
target_seed = 1 # For determining the random target locations
noise_seed = 42 # For determining the random traffic noise locations

N_targets = 50 # The number of targets to be generated
circle_draw_size = 20 # Radius of targets to draw on goal image


'''All functions used'''



def display_img(img, title="Store Map", figsize=(15, 10), cmap='gray', minmax=True):
    plt.figure(figsize=figsize)
    if minmax:
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
    else:
        plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()

def perlin(x, y, seed=0):
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    xi = x.astype(int)
    yi = y.astype(int)
    xf = x - xi
    yf = y - yi
    u = fade(xf)
    v = fade(yf)
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)

def lerp(a, b, x):
    return a + x * (b - a)

def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


def get_traffic_map_and_targets(N, traffic_seed, target_seed, origin=0, scale_factor=5):
    # Load store map
    im_gray = cv2.imread('1150.png', cv2.IMREAD_GRAYSCALE)
    # Binarize
    thresh, im_bw = cv2.threshold(im_gray, 254, 255, cv2.THRESH_BINARY_INV)
    # Fill holes to make impassable areas
    collision_map = im_bw.copy()
    _,contour, hier = cv2.findContours(collision_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(collision_map, [cnt], 0, 255, -1)
    # Edge detect to generate valid targets
    valid_targets = cv2.Laplacian(collision_map, cv2.CV_64F)
    # Generate random targets
    random.seed(target_seed)
    im_targets = im_gray.copy()
    ys, xs = valid_targets.nonzero()
    target_indicies = random.sample(range(len(xs)), N_targets)
    target_xs, target_ys = xs[target_indicies], ys[target_indicies]
    # Generate noise image for traffic
    width, height = len(im_targets[0]), len(im_targets)
    noise_img = generate_noise_image(width, height, seed=noise_seed, origin=origin, scale_factor=scale_factor)
    noise_img = (noise_img - np.min(noise_img)) / (np.max(noise_img) - np.min(noise_img))
    # Merge noise and collisions
    travel_friction = (~collision_map>0).astype(int) * noise_img
    # Enforce collision map infinite condition
    travel_friction[collision_map>0] = np.inf
    # Make targets passable
    for x, y in zip(target_xs, target_ys):
        travel_friction[y][x] = 0
    # Return the friction map and the target points
    return travel_friction, list(zip(target_xs, target_ys))


def generate_noise_image(width, height, origin=0, scale_factor=5, seed=2):
    size = max([width, height])
    scale_x = width / size
    scale_y = height / size
    X = np.linspace(origin, scale_factor * scale_x, width, endpoint=False)
    Y = np.linspace(origin, scale_factor * scale_y, height, endpoint=False)
    x, y = np.meshgrid(X, Y)
    noise_img = perlin(x, y, seed=seed)
    return noise_img

def path_finder_func(base_x,base_next_x,base_y,base_next_y,im_bw):
    im_bw = im_bw / 255
    im_bw = pd.DataFrame(im_bw)
    data = pd.DataFrame(np.logical_xor(im_bw.values, 1).astype(int), columns=im_bw.columns, index=im_bw.index)
    data = data.to_numpy()
    grid = Grid(matrix=data)
    start = grid.node(base_x, base_y)
    end = grid.node(base_next_x, base_next_y)

    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    paths, runs = finder.find_path(start, end, grid)
    print('operations:', runs, 'path length:', len(paths))
    #print(grid.grid_str(path=path, start=start, end=end))
    return(paths)

def dist(x,y):
    return math.hypot(y[0]-x[0],y[1]-x[1])

# ## Critical Hyperparameters
image_map_source_filename = '1150.png' # The store map file location
target_seed = 42 # For determining the random target locations
noise_seed = 42 # For determining the random traffic noise locations
N_targets = 50 # The number of targets to be generated
circle_draw_size = 20 # Radius of targets to draw on goal image

# ## Preparing the Maps
im_gray = cv2.imread('1150.png', cv2.IMREAD_GRAYSCALE)
#display_img(im_gray)

thresh, im_bw = cv2.threshold(im_gray, 254, 255, cv2.THRESH_BINARY_INV)
#display_img(im_bw, title="Store Map Collision")

# ## Generating the Collision Map
collision_map = im_bw.copy()
_,contour, hier = cv2.findContours(collision_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    cv2.drawContours(collision_map, [cnt], 0, 255, -1)
#display_img(collision_map, title="Collision Map")

# ## Determining Target Points
valid_targets = cv2.Laplacian(collision_map, cv2.CV_64F)
#display_img(valid_targets, title="Possible Targets Map")
random.seed(target_seed)
im_targets = im_gray.copy()
ys, xs = valid_targets.nonzero()
target_indicies = random.sample(range(len(xs)), N_targets)
target_xs, target_ys = xs[target_indicies], ys[target_indicies]
for x, y in zip(target_xs, target_ys):
    cv2.circle(im_targets, (x, y), circle_draw_size, (0, 255, 0), -1)
#display_img(im_targets)

x = target_xs
y = target_ys
n = len(x)
g = tuple(zip(x,y))
print(g)
cur = 0
path = [cur]
totalDist = 0
for i in range(1,len(g)):
    dists = [(dist(g[i],p), pi) for (pi,p) in enumerate(g) if pi != i and pi not in path]
    nextDist, cur = min(dists)
    totalDist += nextDist
    path.append(cur)

print(path, totalDist)
final_path = []
for i in range(50):
    # print(data)
    point_number = path[i]
    next_point_number = path[i+1]
    #print(point_number)
    #print(next_point_number)
    base_x = g[point_number][0]
    base_y = g[point_number][1]
    #print(base_x,base_y)
    base_next_x = g[next_point_number][0]
    base_next_y = g[next_point_number][1]
    #print(base_next_x, base_next_y)
    print(base_x,base_next_x)
    print(base_y,base_next_y)
    final_path.append(path_finder_func(base_x,base_next_x,base_y,base_next_y,im_bw))

print(final_path)
#
# def evaluate_path(traffic_map, target_points, path_points):
#     # Confirm the path is contiguous
#     for i in range(0, len(path_points) - 1):
#         if np.abs(path_points[i][0] - path_points[i+1][0]) + np.abs(path_points[i][1] - path_points[i+1][1]) > 1:
#             raise ValueError('Error in path between {0} @ {1} and {2} @ {3}. The path given by path_points has jumps - the path must never move more than 1 square in either the horizontal or vertical direction. Diagonal is not allowed.'.format(i, path_points[i], i+1, path_points[i+1]))
#     # Confirm all points were visited
#     for idx, tp in enumerate(target_points):
#         if tp not in path_points:
#             raise ValueError('Not all target points were visited. Point {0} @ {1} was not visited. Stopping evaluation.'.format(idx, tp))
#     # Determine the score
#     score = 0
#     for pp in path_points:
#         score += traffic_map[pp[1]][pp[0]]
#     return score

map_img, target_points = get_traffic_map_and_targets(50, 42, 42, scale_factor=5)
disp_img = map_img.copy()
for x, y in zip(target_xs, target_ys):
    cv2.circle(disp_img, (x, y), circle_draw_size, (0, 0, 0), -1)
display_img(disp_img*255)
img = map_img

def draw_path(img, path, color=1.0):
    tmp = img.copy()
    cv2.imshow(map_img)
    for p in path:
        tmp[p[1]][p[0]] = color
    cv2.imshow(tmp)

def straight_line_path(start, length):
    path = [start]
    for i in range(length):
        if i%2 == 0:
            path.append((path[-1][0]+1, path[-1][1]))
        else:
            path.append((path[-1][0], path[-1][1]+1))
    return path

# map_img, target_points = get_traffic_map_and_targets(50, 42, 42, scale_factor=5)
# disp_img = map_img.copy()
# for x, y in zip(target_xs, target_ys):
#     cv2.circle(disp_img, (x, y), circle_draw_size, (0, 0, 0), -1)
# display_img(disp_img*255)

# This should fail as not all target points were traversed
#path = straight_line_path((100, 100), 100)

display_img(draw_path(map_img, final_path[0], color=4)*255)

#evaluate_path(map_img, target_points, path)
#

# image = collision_map
# image_rescaled = rescale(image, 1.0, anti_aliasing=False)
# image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)
# im_downscaled = downscale_local_mean(image, (4, 3))
# thresh, im_bw = cv2.threshold(im_downscaled, 254, 255, cv2.THRESH_BINARY_INV)
#
# data = pd.DataFrame(collision_map)
# data = pd.DataFrame(np.logical_xor(data.values,1).astype(int),columns=data.columns,index=data.index)
# data = data.values.tolist()
# im_bw = data
# # for i in range(len(data)):
# #     for j in range(len(data[0])):
# #         if data[i][j] == 0.0:
# #             data[i][j] = 1
# #         elif(data[i][j]==1.0):
# #             data[i][j] = 0
# # im_bw = data
# # base_x = 0
# # base_y = 0
# # start = grid.node(base_x, base_y)
# # base_x+=100
# # base_y+=100
# # end = grid.node(base_x,base_y)
#
# #
# random.seed(target_seed)
# im_targets = im_gray.copy()
# ys, xs = valid_targets.nonzero()
# target_indicies = random.sample(range(len(xs)), N_targets)
# target_xs, target_ys = xs[target_indicies], ys[target_indicies]
#
# print(target_xs,target_ys)
#
# grid = Grid(matrix=im_bw)
#
# for i in range(50):
#     grid = Grid(matrix=im_bw)
#     start = grid.node(2400, 1400)
#     end = grid.node(2460, 1440)
#     path_finder_func(start, end, grid)
#
#
#
#
# #
# #
# # start = grid.node(base_x, base_y)
# # print(start)
# # base_x = 100
# # base_y = 100
# # base_x += 100
# # base_y += 100
# # end = grid.node(base_x,base_y)
# # print(end)
# # path_finder_func(start,end,grid)
# #
# # for i in range(50):
# #     path_finder_func(start, end, grid)
# #

