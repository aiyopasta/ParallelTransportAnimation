# Conclusion: Really, really slow. Even for few triangles. There's likely ways of making it faster, but that's kinda
# boring to me. I'd rather just start animating stuff now. So let's leave this aside for now, and just get the main
# animations in. Even if you see tkinter-like animations for debugging, that's fine. You can add the fancy rasterization
# at export time if you want (it'll be like 1 second per frame for exporting, but who cares).
#
# Anyways, good job! You still remember stuff from 560 lol.

import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numba as nb
from numba.np.extensions import cross2d
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
@nb.njit
def A(val):
    return np.array([val[0] + width / 2, -val[1] + height / 2])


# 2. Actual perspective projection function
# TODO 1: Numba accelerate
# TODO 2: Give it a third output, which is just the z coordinate wrt the camera (for z-testing).
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


# 4. Barycentric coordinates
@nb.njit
def barycoords(point, positions):
    assert len(point) == 2 and len(positions) == 3
    xy0, xy1, xy2 = positions[0][:-1], positions[1][:-1], positions[2][:-1]
    total_area = np.abs(cross2d(xy1 - xy0, xy2 - xy0)) / 2
    coords = np.array([0., 0., 0.])
    for idx in range(3):
        p1, p2 = positions[(idx + 1) % 3][:-1], positions[(idx + 2) % 3][:-1]
        area = np.abs(cross2d(p2 - p1, point - p1)) / 2
        coords[idx] = area / total_area
    return coords


@nb.njit
def interpolate_zdepth(point, positions):
    assert len(point) == 2 and len(positions) == 3
    coords = barycoords(point, positions)
    reciprocal = 0.
    for i in range(3):
        assert positions[i][-1] >= 1  # smallest z distance is 1 (think about how perspective divide works)
        reciprocal += coords[i] * (1. / positions[i][-1])

    return 1. / reciprocal


# 6. Perspective correct interpolation for arbitrary 3-vector attribute
@nb.njit
def interpolate_attributes(point, positions, attributes):
    assert len(point) == 2 and len(positions) == len(attributes) == 3
    coords = barycoords(point, positions)
    summation = np.array([0., 0., 0.])   # we'll only ever have to interpolate 3-vectors, right? RIGHT?
    for i in range(3):
        assert positions[i][-1] >= 1  # smallest z distance is 1 (think about how perspective divide works)
        summation += coords[i] * (1. / positions[i][-1]) * attributes[i]

    return summation * interpolate_zdepth(point, positions)


# ___ For testing the above functions
# point_ = np.array([10., 10.])
# positions_ = np.array([np.array([0., 0., 1.]), np.array([100., 0., 1.]), np.array([0., 100., 20.])])
# attributes_ = np.array([np.array([255., 255., 255.]), np.array([255., 255., 255.]), np.array([0., 0., 0.])])
# print(interpolate_zdepth(point_, positions_))



# Vertex transformation functions
@nb.njit
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
        self.normal = np.array(normal) if normal is not None else np.array([1.,1.,1.])
        self.color = np.array(color) if color is not None else np.array([1.,1.,1.]) * 255.0
        self.id = Vertex.new_id
        Vertex.new_id += 1

    def raw_data(self, proj=True):
        position = perspective_project(self.position) if proj else self.position
        return position, self.normal, self.color, self.id


class Triangle:
    new_id = 0

    def __init__(self, vertices):
        assert len(vertices) == 3
        self.verts = vertices   # list of Vertex objects
        self.id = Triangle.new_id
        Triangle.new_id += 1

    # Note: NOT A'D!
    def raw_data(self, proj=True):
        positions, normals, colors, IDs = [], [], [], []
        for v in self.verts:
            pos, nor, col, ID = v.raw_data(proj)
            positions.append(pos)
            normals.append(nor)
            colors.append(col)
            IDs.append(ID)

        return np.array(positions), np.array(normals), np.array(colors), np.array(IDs), self.id


# And a mesh class... just cleaner.
class Mesh:
    new_id = 0

    def __init__(self, tri_list):
        self.facets = tri_list  # list of Triangle objects
        self.id = Mesh.new_id
        Mesh.new_id += 1

    # For Numba functions. Note: NOT A'D!
    def raw_data(self, proj=True):
        all_positions, all_normals, all_colors, all_IDs, all_triangle_IDs = [], [], [], [], []
        for facet in self.facets:
            positions, normals, colors, IDs, triangle_id = facet.raw_data(proj)
            all_positions.append(positions)
            all_normals.append(normals)
            all_colors.append(colors)
            all_IDs.append(IDs)
            all_triangle_IDs.append(triangle_id)

        return np.array(all_positions), np.array(all_normals), np.array(all_colors), np.array(all_IDs), np.array(all_triangle_IDs), self.id


# Let's test all this out by outputting the same 3 triangles as before but in this new format.
# We can also shade in the triangles differently based on interpolation.
# TEST 1 —— Single triangle, NO perspective projection
v1 = Vertex(position=np.array([0., 0., 1.]), color=np.array([255., 0., 0.]))   # Note: Here, third positional coordinate is the z-value. Ideally, this would be returned by the perspective projection function.
v2 = Vertex(position=np.array([100., 0., 1.]), color=np.array([0., 255., 0.]))
v3 = Vertex(position=np.array([0., 100., 2.]), color=np.array([0., 0., 255.]))
v4 = Vertex(position=np.array([-100., 0., 1.]), color=np.array([255., 0., 0.]))
v5 = Vertex(position=np.array([50., -40., 4.]), color=np.array([0., 0., 255.]))
v6 = Vertex(position=np.array([0., 200., 2.]), color=np.array([0., 255., 0.]))
triangle = Triangle(np.array([v1, v2, v3]))
triangle2 = Triangle(np.array([v4, v5, v6]))
mesh = Mesh(np.array([triangle, triangle2]))

# Z stuff
z_buffer = np.ones([width, height]) * float('inf')

# For FPS
clock = pygame.time.Clock()

@nb.njit
def rasterize(positions, normals, colors, z_buff):   # TODO: Ideally: mesh_data, z_buff. mesh_data has triangles, each of which have vertices that each have a color, normal, etc.
    '''
        NOTE: Positions of vertices of triangles must be NON-A'D!!!!
    '''
    RGB = np.zeros((3, height, width), dtype=np.uint8)

    # Iterate over each triangle. Remember, each 'pos' contains 3 vertex positions.
    for k, pos in enumerate(positions):
        # Compute bounding box
        xrange = [min(min(pos[0][0], pos[1][0]), pos[2][0]), max(max(pos[0][0], pos[1][0]), pos[2][0])]
        yrange = [min(min(pos[0][1], pos[1][1]), pos[2][1]), max(max(pos[0][1], pos[1][1]), pos[2][1])]
        # Check if non-degenerate
        if abs(xrange[1] - xrange[0]) > 1 and abs(yrange[1] - yrange[0]) > 1:
            # Iterate row by row over bounding box and find out range of columns to shade
            for row in range(int(yrange[0]), int(yrange[1]), 1):
                # By default, don't shade any
                xleft, xright = xrange[1], xrange[0]
                # Let's see which triangle edges the row intersects, if any to figure it out.
                r0_row = np.array([xrange[0], row])
                v_row = np.array([xrange[1], row]) - r0_row
                for i in range(3):
                    p0, p1 = pos[i][:-1], pos[(i + 1) % 3][:-1] # 3rd entry is the z-value (not needed here)
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

                # Actually shade in the correct pixels (columns) for this row
                for col in range(int(xleft), int(xright)):
                    # Compute interpolated z coordinate for this fragment
                    raw_point = np.array([col, row])
                    z_depth = interpolate_zdepth(raw_point, pos)
                    # Convert to A'd coordinates (as that's what the Z-buffer knows about).
                    xy = A(raw_point)
                    true_row, true_col = int(xy[1]), int(xy[0])
                    # Overwrite iff this z value is closer than one in buffer.
                    if z_depth < z_buff[true_row][true_col]:
                        z_buff[true_row][true_col] = z_depth
                        # Get interpolated color
                        interp_color = interpolate_attributes(raw_point, pos, colors[k])
                        for channel in range(3):  # store separate RGB
                            RGB[channel][true_row][true_col] = interp_color[channel]

                        # window.set_at(xy, colors[k])

    return RGB


# Task 2:
def main():
    global z_buffer, mesh, clock

    # Game loop
    count = 0
    run = True
    while run:
        # Reset everything.
        window.fill((0, 0, 0))
        z_buffer = np.ones([width, height]) * float('inf')

        RGB = rasterize(*mesh.raw_data(proj=False)[:3], z_buffer)
        RGB = np.transpose(RGB, (2, 1, 0))
        pygame.surfarray.blit_array(window, RGB)


        # We handle keys pressed inside the gameloop in PyGame
        keys_pressed = pygame.key.get_pressed()
        # do stuff....

        # End frame and update everything.
        pygame.display.update()
        fps = clock.tick_busy_loop()
        count += 1
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
