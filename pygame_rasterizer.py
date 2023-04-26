import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from numba import njit
import copy

# Pygame + gameloop setup
width = 1700   # 800
height = 1000  # 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("3D Projection + Rasterization")
np.set_printoptions(suppress=True)

# Camera parameters (rho = distance from origin, phi = angle from world's +z-axis, theta = angle from world's +x-axis)
cam_rho, cam_theta, cam_phi = 1000., np.pi/4, np.pi/4  # Rho is the distance from world origin to near clipping plane
v_rho, v_theta, v_phi = 0, 0, 0
cam_focus = 1000.  # Distance from near clipping plane to eye


# Helper functions
# 1. Coordinate Shift
@njit
def A(val):
    return np.array([val[0] + width / 2, -val[1] + height / 2])


# 2. Actual perspective projection function
# TODO: Numba accelerate
# RECALL THIS GIVES RAW, UN-A'D OUTPUT!
def world_to_plane(v, rho, theta, phi, focus):
    '''
        Converts from point in 3D to its 2D perspective projection, based on location of camera.

        v: vector in R^3 to convert.
    '''
    # Radial distance to eye from world's origin.
    eye_rho = rho + focus

    # Vector math from geometric computation (worked out on white board, check iCloud for possible picture)
    eye_to_origin = -np.array([eye_rho * np.sin(phi) * np.cos(theta),
                               eye_rho * np.sin(phi) * np.sin(theta), eye_rho * np.cos(phi)])

    eye_to_ei = eye_to_origin + v
    origin_to_P = np.array(
        [rho * np.sin(phi) * np.cos(theta), rho * np.sin(phi) * np.sin(theta), rho * np.cos(phi)])

    # Formula for intersecting t: t = (n•(a-b)) / (n•v)
    t = np.dot(eye_to_origin, origin_to_P - v) / np.dot(eye_to_origin, eye_to_ei)
    r_t = v + (t * eye_to_ei)

    # Location of image coords in terms of world coordinates.
    tile_center_world = -origin_to_P + r_t

    # Spherical basis vectors
    theta_hat = np.array([-np.sin(theta), np.cos(theta), 0])
    phi_hat = -np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), -np.sin(phi)])

    # Actual transformed 2D coords
    tile_center = np.array([np.dot(tile_center_world, theta_hat), np.dot(tile_center_world, phi_hat)])

    return tile_center


# 3. Wrapper function for perspective projection
def perspective_project(v):
    global cam_rho, cam_theta, cam_phi, cam_focus
    return world_to_plane(v, cam_rho, cam_theta, cam_phi, cam_focus)


# Vertex transformation functions
@njit
def rot(vertex, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R.dot(vertex)

# Task 1: Rasterize a single triangle on the screen, and measure performance. —— Complete
# tri = [np.array([0., 0.]),
#        np.array([100., 0.]),
#        np.array([0., 100.])]  # counter-clockwise

# Task 2: Setup a z-buffer and rasterize many triangles on top of one another
# tri1 = [np.array([0., 0.]),
#         np.array([100., 0.]),
#         np.array([0., 100.])]  # counter-clockwise
# tri2 = [np.array([25., 25.]),
#         np.array([100.+25., 0.+25.]),
#         np.array([0.+25, 100.+25.])]  # counter-clockwise
# tri3 = [np.array([0.+50, 0.+10.]),
#         np.array([100.+50, 0.+10.]),
#         np.array([0.+50, 100.+10.])]  # counter-clockwise
# triangles = [tri1, tri2, tri3]
# # for i in range(50):
# #     offset = i * np.array([1, 1]) * 30
# #     tri_new = [vert + offset for vert in tri1]
# #     triangles.append(tri_new)
# triangles = np.array(triangles)
# print(f'Total {len(triangles)} triangles')
# z_values = np.array([100, 50, 10])  # TEST 1 —— Every triangle exists in axis-aligned plane.
# cols = np.array([[255, 0, 0], [0, 0, 255], [0, 255, 0]])

# Task 3: Implement perspective projection + rasterization of a single triangle.
# Let's make a triangle class now....
class Vertex:
    new_id = 0

    def __init__(self, position, color=None, normal=None):
        self.position = np.array(position)
        self.normal = np.array(normal)
        self.color = color if color is None else np.array([1.,1.,1.]) * 255.0
        self.id = Vertex.new_id
        Vertex.new_id += 1

    def raw_data(self, proj=True):
        position = perspective_project(self.position) if proj else self.position
        return np.array([position, self.normal, self.color, self.id])


class Triangle:
    new_id = 0

    def __init__(self, vertices, vertex_cols):
        assert len(vertices) == 3
        self.verts = vertices   # list of Vertex objects
        self.cols = vertex_cols
        self.id = Triangle.new_id
        Triangle.new_id += 1

    # Note: NOT A'D!
    def raw_data(self, proj=True):
        return np.array([v.raw_data(proj) for v in self.verts])


# And a mesh class... just cleaner.
class Mesh:
    new_id = 0

    def __init__(self, tri_list):
        self.facets = tri_list  # list of Triangle objects
        self.id = Mesh.new_id
        Mesh.new_id += 1

    # For Numba functions. Note: NOT A'D!
    def raw_data(self, proj=True):
        return np.array([face.raw_data(proj) for face in self.facets])

# Let's test all this out by outputting the same 3 triangles as before but in this new format.
# We can also shade in the triangles differently based on interpolation.
# TEST 1 —— NO perspective projection




z_buffer = np.ones([width, height]) * float('inf')

# For FPS
clock = pygame.time.Clock()

@njit
def rasterize(tri_list, z_vals, colors, z_buff):   # TODO: Ideally: mesh_data, z_buff. mesh_data has triangles, each of which have vertices that each have a color, normal, etc.
    '''
        NOTE: Positions of vertices of triangles must be NON-A'D!!!!
    '''
    RGB = np.zeros((3, height, width), dtype=np.uint8)
    # 1. Iterate over list of triangles
    # 2. When you're shading each, compare to the z_buff + update it if necessary
    for k, tri in enumerate(tri_list):
        xrange = [min(min(tri[0][0], tri[1][0]), tri[2][0]), max(max(tri[0][0], tri[1][0]), tri[2][0])]
        yrange = [min(min(tri[0][1], tri[1][1]), tri[2][1]), max(max(tri[0][1], tri[1][1]), tri[2][1])]
        if abs(xrange[1] - xrange[0]) > 1 and abs(yrange[1] - yrange[0]) > 1:
            for row in range(int(yrange[0]), int(yrange[1]), 1):
                xleft, xright = xrange[1], xrange[
                    0]  # initialize as left > right, i.e. "do NOT shade any pixels in row"
                r0_row = np.array([xrange[0], row])
                v_row = np.array([xrange[1], row]) - r0_row
                for i in range(3):
                    p0, p1 = tri[i], tri[(i + 1) % 3]
                    r0_edge = p0
                    v_edge = p1 - p0
                    # Normal, non-degenerate intersection case
                    if v_edge[1] != 0:
                        M = np.vstack((v_row, -v_edge)).T
                        b = r0_edge - r0_row
                        tau = np.linalg.inv(M).dot(b)
                        # Check if line SEGMENTS actually intersect (within box), and update xleft/right accordingly.
                        if 0 <= tau[0] <= 1 and 0 <= tau[1] <= 1:
                            x_intersect = ((tau[0] * v_row) + r0_row)[0]
                            xleft = min(xleft, x_intersect)
                            xright = max(xright, x_intersect)
                    # Edge parallel with row AND overlapping
                    elif r0_row[1] == r0_edge[1]:
                        # Moreover check if the line SEGMENTS actually intersect
                        left_vert_x, right_vert_x = min(p0[0], p1[0]), max(p0[0], p1[0])
                        if xrange[0] <= left_vert_x <= xrange[1] or left_vert_x <= xrange[0] <= right_vert_x:
                            xleft, xright = xright, xleft
                            break

                # Actually shade in all the pixels
                for col in range(int(xleft), int(xright)):
                    # Overwrite iff this z value is closer than one in buffer.
                    xy = A(np.array([col, row]))
                    true_row, true_col = int(xy[1]), int(xy[0])
                    # TODO: The actual z value for this fragment will have to be perspective-correctly interpolated.
                    if z_vals[k] < z_buff[true_row][true_col]:
                        z_buff[true_row][true_col] = z_vals[k]
                        for channel in range(3):  # store separate RGB
                            RGB[channel][true_row][true_col] = colors[k][channel]

                        # window.set_at(xy, colors[k])

    return RGB


# Task 2:
def main():
    global z_buffer, triangles, z_values, cols, clock  #, tri

    # Game loop
    count = 0
    run = True
    while run:
        window.fill((0, 0, 0))
        count += 1
        z_buffer = np.ones([width, height]) * float('inf')  # reset each frame

        for i, tri in enumerate(triangles):
            p0 = triangles[i][0]
            for j, vert in enumerate(tri):
                triangles[i][j] = rot(triangles[i][j] - p0, np.pi / 100) + p0

        RGB = rasterize(triangles, z_values, cols, z_buffer)
        RGB = np.transpose(RGB, (2, 1, 0))
        pygame.surfarray.blit_array(window, RGB)


        # We handle keys pressed inside the gameloop in PyGame
        keys_pressed = pygame.key.get_pressed()
        # do stuff....

        # End run (1. Tick clock, 2. Save the frame, and 3. Detect if window closed)
        pygame.display.update()
        fps = clock.tick_busy_loop()
        print('FPS', int(1000/fps))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

def main_old():
    global tri

    for i in range(3):
        print(tri[i])

    print()
    print()



    # Game loop
    count = 0
    run = True
    while run:
        window.fill((0, 0, 0))
        count += 1

        # # TEST 1: ROTATION —— PASSED
        # for i in range(3):
        #     tri[i] = rot(tri[i], np.pi / 100)
        #
        # # TEST 2: TRIANGLE MANIA —— PASSED
        # tri[0] += np.array([1, 1])
        # tri[1] += np.array([-1, 1])

        # TEST 3: FLIP FLOP BOOGALOO —— PASSED
        # tri[2] *= 0.99 * np.power(-1, count)
        # tri[1] -= np.array([1, 1])

        # TEST 4: COLLINEAR POINTS —— GOOD BOY!
        tri = [np.array([0., 0.]),
               np.array([30., 30.]),
               np.array([100., 100.])]  # counter-clockwise

        tri[1] += np.array([-1, 1]) * count

        pygame.draw.polygon(window, (0, 255, 0), [A(vert) for vert in tri])

        # 1. Compute bounding box
        # 2. For each row of pixels inside the box, find where that row (fictitious line) intersects with edges of triangle
        # 3. For every pixel within the intersection range, shade it in properly using whatever interpolation method or whatever.
        xrange = [min(min(tri[0][0], tri[1][0]), tri[2][0]), max(max(tri[0][0], tri[1][0]), tri[2][0])]
        yrange = [min(min(tri[0][1], tri[1][1]), tri[2][1]), max(max(tri[0][1], tri[1][1]), tri[2][1])]
        if abs(xrange[1] - xrange[0]) > 1 and abs(yrange[1] - yrange[0]) > 1:
            for row in range(int(yrange[0]), int(yrange[1]), 1):
                xleft, xright = xrange[1], xrange[0]  # initialize as left > right, i.e. "do NOT shade any pixels in row"
                r0_row = np.array([xrange[0], row])
                v_row = np.array([xrange[1], row]) - r0_row
                for i in range(3):
                    p0, p1 = tri[i], tri[(i+1) % 3]
                    r0_edge = p0
                    v_edge = p1 - p0
                    # Normal, non-degenerate intersection case
                    if v_edge[1] != 0:
                        M = np.vstack([v_row, -v_edge]).T
                        b = r0_edge - r0_row
                        tau = np.linalg.inv(M).dot(b)
                        # Check if line SEGMENTS actually intersect (within box), and update xleft/right accordingly.
                        if 0 <= tau[0] <= 1 and 0 <= tau[1] <= 1:
                            x_intersect = ((tau[0] * v_row) + r0_row)[0]
                            xleft = min(xleft, x_intersect)
                            xright = max(xright, x_intersect)
                    # Edge parallel with row AND overlapping
                    elif r0_row[1] == r0_edge[1]:
                        # Moreover check if the line SEGMENTS actually intersect
                        left_vert_x, right_vert_x = min(p0[0], p1[0]), max(p0[0], p1[0])
                        if xrange[0] <= left_vert_x <= xrange[1] or left_vert_x <= xrange[0] <= right_vert_x:
                            xleft, xright = xright, xleft
                            break

                # Actually shade in all the pixels
                for col in range(int(xleft), int(xright)):
                    xy = A(np.array([col, row])).astype(int)
                    window.set_at(xy, (255, 0, 0))


        # We handle keys pressed inside the gameloop in PyGame
        keys_pressed = pygame.key.get_pressed()
        # do stuff....


        # End run (1. Tick clock, 2. Save the frame, and 3. Detect if window closed)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()


if __name__ == "__main__":
    main()
