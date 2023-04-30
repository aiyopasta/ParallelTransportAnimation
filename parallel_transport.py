import copy

import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from playsound import playsound
from functools import partial

# Save the animation? TODO: Make sure you're saving to correct destination!!
save_anim = True

# Pygame + gameloop setup
width = 800
height = 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Parallel Transport Animations")
pygame.init()


# Coordinate Shift
def A(val):
    return np.array([val[0] + width / 2, -val[1] + height / 2])

def A_inv(val):
    global width, height
    return np.array([val[0] - width / 2, -(val[1] - height / 2)])


def A_many(vals):
    return [A(v) for v in vals]   # Galaxy brain function


# Camera parameters (rho = distance from origin, phi = angle from world's +z-axis, theta = angle from world's +x-axis)
rho, theta, phi = 1000., -np.pi/2, 0  # Rho is the distance from world origin to near clipping plane
v_rho, v_theta, v_phi = 0, 0, 0
focus = 1000000.  # Distance from near clipping plane to eye


# Normal perspective project
def world_to_plane(v):
    '''
        Converts from point in 3D to its 2D perspective projection, based on location of camera.

        v: vector in R^3 to convert.
        NOTE: Here, we do NOT "A" the final output.
    '''
    # Camera params
    global rho, theta, phi, focus

    # Radial distance to eye from world's origin.
    eye_rho = rho + focus

    # Vector math from geometric computation (worked out on white board, check iCloud for possible picture)
    eye_to_origin = -np.array([eye_rho * np.sin(phi) * np.cos(theta),
                               eye_rho * np.sin(phi) * np.sin(theta), eye_rho * np.cos(phi)])

    eye_to_ei = eye_to_origin + v
    origin_to_P = np.array(
        [rho * np.sin(phi) * np.cos(theta), rho * np.sin(phi) * np.sin(theta), rho * np.cos(phi)])

    # Formula for intersecting t: t = (n•(a-b)) / (n•v)
    tau = np.dot(eye_to_origin, origin_to_P - v) / np.dot(eye_to_origin, eye_to_ei)
    r_t = v + (tau * eye_to_ei)

    # Location of image coords in terms of world coordinates.
    tile_center_world = -origin_to_P + r_t

    # Spherical basis vectors
    theta_hat = np.array([-np.sin(theta), np.cos(theta), 0])
    phi_hat = -np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), -np.sin(phi)])

    # Actual transformed 2D coords
    tile_center = np.array([np.dot(tile_center_world, theta_hat), np.dot(tile_center_world, phi_hat)])

    return tile_center


def world_to_plane_many(pts):
    return np.array([world_to_plane(v) for v in pts])  # galaxy brain function


def frame_shift2d(pt, offset, radians=0):
    M = np.array([[np.cos(radians), -np.sin(radians)],
                  [np.sin(radians), np.cos(radians)]])
    return M @ (pt + offset)  #(M @ pt) + offset


# 2D reference frame shift
def frame_shift2d_many(pts, offset, radians=0):
    return [frame_shift2d(pt, offset, radians) for pt in pts]


# Keyframe / timing params
FPS = 60
t = 18
dt = 0.01    # i.e., 1 frame corresponds to +0.01 in parameter space = 0.01 * FPS = +0.6 per second (assuming 60 FPS)

keys = [0,    # Keyframe 0. Draw origin + gridlines.
        1,    # Keyframe 1. Pan to the location of our traveler.
        2,    # Keyframe 2. Traveler's journey
        8,    # Keyframe 3. Traveler's journey, again.
        14,   # Keyframe 4. Traveler's journey, yet again.
        18,   # Keyframe 5. Traveler's journey, but it's a square (zoomed out)
        19,   # Keyframe 6. Undo traveler's journey.  TODO
        20,   # Keyframe 7. Fade out portion of the plane and draw portals.
        22,   # Keyframe 8. Show new path going through the portals.
        24,   # Keyframe 9. 2D to cone transformation with the path on it too.
        30    # DONE.
        ]

# keys.extend([keys[-1] + 10, keys[-1] + (10 * 2)])  # Placeholder + done


# Helper functions
def squash(t_, intervals=None):
    global keys
    if intervals is None:
        intervals = keys
    for i in range(len(intervals) - 1):
        if intervals[i] <= t_ < intervals[i + 1]:
            return (t_ - intervals[i]) / (intervals[i + 1] - intervals[i]), i

    return intervals[-1], len(intervals) - 2


# Specific case of the squash. We squash t into equally sized intervals.
def squash2(t_, n_intervals=1):
    intervals = [float(i) / n_intervals for i in range(n_intervals + 1)]
    return squash(t_, intervals)


def slash(t_, new_end=0.5):
    return t_ / new_end if t_ < new_end else 1.0


def rot_mat(radians):
    M = np.array([[np.cos(radians), -np.sin(radians), 0],
                  [np.sin(radians), np.cos(radians), 0],
                  [0, 0, 1]])
    return M


# Easing functions.
# TODO: Add more!
def ease_inout(t_):
    return np.power(t_, 2) / (np.power(t_, 2) + np.power(1 - t_, 2))


def ease_out(t_):
    return 1.0 - np.power(1.0 - t_, 2.0)


# From ChatGPT
def rgb_to_hex(rgb):
    return '#' + ''.join(['%02x' % int(val) for val in rgb])


# Parametric shapes
def line(u, start, fin):
    return ((1 - u) * start) + (u * fin)


def circle(u, radius, center_x=0, center_y=0):
    tau = 2 * np.pi * u
    return (radius * np.array([np.cos(tau), np.sin(tau)])) + np.array([center_x, center_y])


def cosine(u, offset_x, offset_y, scale=30, cycles=3):
    tau = u * 2.0 * np.pi * scale * cycles  # period = 2pi * scale
    return np.array([tau + offset_x, (scale * np.cos(tau / scale)) + offset_y])


# TODO: Make it fancier
def traveler(u, radius, center_x=0, center_y=0, jiggle=1):
    return circle(1, radius, center_x, center_y)


# Shape sampling functions
def get_circle_pts(u, radius, center_x, center_y, du=0.001):
    pts = []
    for u_ in np.arange(0, u + du, du):
        pts.append(circle(u_, radius, center_x, center_y))
    return pts


def get_many_line_pts(u, start, fin, du=0.001):
    pts = []
    for u_ in np.arange(0, u + du, du):
        pts.append(line(u_, start, fin))
    return pts


def get_cosine_pts(u, offset_x, offset_y, scale=30, cycles=3, du=0.001):
    pts = []
    for u_ in np.arange(0, u + du, du):
        pts.append(cosine(u_, offset_x, offset_y, scale=scale, cycles=cycles))
    return pts


# TODO: Make it fancier (give it eyes, make it jiggle, etc.)
def get_traveler_pts(u, radius, center_x=0, center_y=0, jiggle=1, du=0.001):
    pts = []
    for u_ in np.arange(0, u + du, du):
        pts.append(circle(u_, radius, center_x, center_y))
    return pts


# 2D plane to cone transformation
def to_cone(u, point, cutout_interval=(np.pi, 3*np.pi/2)):
    assert len(point) == 3 and point[2] == 0  # must be a 3d point on the xy plane
    oneOverTwoPi = 1.0 / (2.0 * np.pi)
    # 0. Basic params
    cutout0, cutout1 = cutout_interval[0], cutout_interval[1]
    assert 0 <= cutout0 <= cutout1 < 2 * np.pi
    cutout_range = abs(cutout1 - cutout0)
    # 1. Convert to 2D polar coordinates (r, theta)
    radius_, theta_ = np.linalg.norm(point), np.arctan2(point[1], point[0]) % (2*np.pi)
    assert 0 <= abs(theta_) <= 2 * np.pi   # abs is there because sometimes it's "-0.0" 🙄
    assert theta_ <= cutout0 or theta_ >= cutout1
    # 2. Bijectively map (r, theta) to a particular cylindrical coordinate (rho, beta, z)
    rho_ = (1.0 - (cutout_range * oneOverTwoPi)) * radius_
    z_ = -np.sqrt(np.power(radius_, 2.0) - np.power(rho_, 2.0)) + 500
    #  Given x, y, and z, with x,z non-zero, we want x' and z' such that:
    #  total := x + y + z  =  x' + z'
    #  x'/x = z'/z  <=>  x'z = z'x  <=>  z' = x'z/x  =>  x' = total / (1 + z/x)  =>  z' = total - x'
    #  Here, x := cutout0, z := 2pi - cutout1, and y := cutout_range.
    #  And x' := cutout0 AFTER transformation, and z' := 2pi - cutout0 AFTER transformation.
    x_prime = (2 * np.pi / (1.0 + (((2 * np.pi) - cutout1) / cutout0)))
    z_prime = (2 * np.pi) - x_prime
    beta_ = (theta_ / cutout0) * x_prime
    if theta_ >= cutout1:
        beta_ = (((theta_ - cutout1) / ((2 * np.pi) - cutout1)) * z_prime) + x_prime
    # 3. Convert cylindrical coords back to 3D cartesian
    cone_point = np.array([rho_ * np.cos(beta_), rho_ * np.sin(beta_), z_])
    # 4. Interpolate between original point and cone point
    # TODO: Change interpolation to be between cylindrical coords, in order to make the
    # TODO: interpolation between beta angles FASTER than the other ones to give a better "FOLDING" effect.
    return ((1 - u) * point) + (u * cone_point)


def to_cone_many(u, pts, cutout_interval=(np.pi, 3*np.pi/2)):
    return np.array([to_cone(u, pt, cutout_interval) for pt in pts])


# Transform path (sequence of points) into one which obeys sectorial portals
def through_portals(path_points, cutout_interval=(np.pi, 3*np.pi/2)):
    new_path_points = copy.copy(path_points)  # to return
    assert 0. <= cutout_interval[0] <= cutout_interval[1] < 2. * np.pi
    R = np.eye(3)  # rotation matrix to keep track of
    cutout_angle = cutout_interval[1] - cutout_interval[0]
    cut_indices = [0]
    for i, pt in enumerate(path_points):
        # Handle i == 0 case
        transf_pt = R @ pt
        arg = np.arctan2(transf_pt[1], transf_pt[0]) % (2 * np.pi)
        in_void = cutout_interval[0] <= arg <= cutout_interval[1]
        if i == 0:
            # don't start with a point in the void
            assert not (i == 0 and in_void)
            continue

        # General case. If encountered new portal, update the R matrix to account for it.
        prev_point = R @ path_points[i - 1]
        prev_arg = np.arctan2(prev_point[1], prev_point[0]) % (2 * np.pi)
        prev_in_void = cutout_interval[0] <= prev_arg <= cutout_interval[1]
        newly_entered_void = in_void and not prev_in_void
        if newly_entered_void:
            # If entered through [0] opening then counter-clockwise rotation (+1 sign), otherwise if entered
            # through [1] opening then clockwise rotation (-1 sign).
            sign = +1 if prev_arg <= cutout_interval[0] else -1  # certain assumption about cutout made here!!
            R = np.dot(rot_mat(sign * cutout_angle), R)
            cut_indices.append(i)

        # Actually transform the current point through new portal
        new_path_points[i] = R @ path_points[i]

    cut_indices.append(len(path_points))
    return new_path_points, cut_indices


# Keyhandling
def handle_keys(keys_pressed):
    global rho, theta, phi, focus
    m = 30
    drho, dphi, dtheta, dfocus = 10, np.pi/m, np.pi/m, 10

    if keys_pressed[pygame.K_w]:
        phi -= dphi
    if keys_pressed[pygame.K_a]:
        theta -= dtheta
    if keys_pressed[pygame.K_s]:
        phi += dphi
    if keys_pressed[pygame.K_d]:
        theta += dtheta
    if keys_pressed[pygame.K_p]:
        rho -= drho
    if keys_pressed[pygame.K_o]:
        rho += drho
    if keys_pressed[pygame.K_k]:
        focus -= dfocus
    if keys_pressed[pygame.K_l]:
        focus += dfocus


# def speedy_key_pressed(event):
#     global v_rho, v_theta, v_phi, focus, w
#
#     max_clicks = 10
#     m = 800
#     d2rho, d2phi, d2theta, dfocus = 3, np.pi / m, np.pi / m, 10
#     if event.char == 'a':
#         v_theta = max(v_theta - d2theta, -d2theta*max_clicks)
#
#     if event.char == 'd':
#         v_theta = min(v_theta + d2theta, d2theta*max_clicks)
#
#     if event.char == 'w':
#         v_phi = max(v_phi - d2phi, -d2phi*max_clicks)
#
#     if event.char == 's':
#         v_phi = min(v_phi + d2phi, d2phi*max_clicks)
#
#     if event.char == 'p':
#         v_rho = max(v_rho - d2rho, -d2rho*max_clicks/2)
#
#     if event.char == 'o':
#         v_rho = min(v_rho + d2rho, d2rho*max_clicks/2)
#
#     if event.char == 'k':
#         focus -= dfocus
#
#     if event.char == 'l':
#         focus += dfocus
#
#     # Change mode
#     if event.char == 'm':
#         v_rho, v_theta, v_phi = 0, 0, 0
#         w.bind("<KeyPress>", vanilla_key_pressed)



# Additional vars / knobs
play_music = False
white = np.array([255, 255, 255])
traveler_blue = np.array([138, 204, 255])
trace_line_col = np.array([255, 23, 89])
black = np.array([0, 0, 0])


def main():
    global t, dt, keys, FPS, rho, phi, theta, focus, save_anim, play_music, white, black, trace_line_col

    # Pre-animation setup
    clock = pygame.time.Clock()
    run = True

    # Animation saving setup
    path_to_save = '/Users/adityaabhyankar/Desktop/Programming/ParallelTransportAnimation/pygame_output'
    if save_anim:
        for filename in os.listdir(path_to_save):
            # Check if the file name follows the required format
            b1 = filename.startswith("frame") and filename.endswith(".png")
            b2 = filename.startswith("output.mp4")
            if b1 or b2:
                os.remove(os.path.join(path_to_save, filename))
                print('Deleted frame ' + filename)

    # Text data
    font = pygame.font.SysFont("Avenir Next", 70)
    text = font.render("START", True, (252, 3, 90))

    # Game loop
    count = 0
    while run:
        # Reset stuff
        window.fill((0, 0, 0))

        # Animation!
        u, frame = squash(t)
        # Keyframe 0 —— Draw origin + gridlines.
        if frame == 0:
            tau = slash(u, 0.2)
            u = ease_inout(u)
            frame_offset, frame_angle = np.array([0,0]), 0
            shift_many = partial(frame_shift2d_many, offset=frame_offset, radians=frame_angle)
            shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
            # Vertical Lines
            spacing = 100
            grid_width, grid_height = width * 5, height * 5
            for x in np.arange(-grid_width/2, grid_width/2, spacing):
                start_y, finish_y = grid_height/2, -grid_height/2
                start_pt, finish_pt = world_to_plane(np.array([x, start_y, 0])), world_to_plane(np.array([x, finish_y, 0]))
                middle_pt = line(u, start_pt, finish_pt)
                pygame.draw.line(window, white * 0.5, *A_many(shift_many([start_pt, middle_pt])), width=2)

            # Horizontal Lines
            for y in np.arange(-grid_height/2, grid_height/2, spacing):
                start_x, finish_x = -grid_width/2, grid_width/2
                start_pt, finish_pt = world_to_plane(np.array([start_x, y, 0])), world_to_plane(np.array([finish_x, y, 0]))
                middle_pt = line(u, start_pt, finish_pt)
                pygame.draw.line(window, white * 0.5, *A_many(shift_many([start_pt, middle_pt])), width=2)

            # Origin
            if tau < 1.0:
                radius = 10
                center3d = np.array([0,0,0])
                pts = get_circle_pts(tau, radius, *world_to_plane(center3d), du=0.001)
                if len(pts) > 2:
                    pygame.draw.lines(window, white, False, A_many(shift_many(pts)), width=5)
                    pygame.draw.polygon(window, white, A_many(shift_many(pts)), 0)
            else:
                radius = 13
                center2d = A(shift(world_to_plane(np.array([0, 0, 0]))))
                pygame.draw.circle(window, white, center2d, radius, width=0)


        # Keyframe 1 —— Pan to the location of our traveler.
        if frame == 1:
            # The frame shift
            u = ease_inout(u)
            spacing = 100
            frame_offset, frame_angle = u * (spacing * 5) * np.array([-1, 1]), 0
            shift_many = partial(frame_shift2d_many, offset=frame_offset, radians=frame_angle)
            shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
            # Vertical Lines
            grid_width, grid_height = width * 5, height * 5
            for x in np.arange(-grid_width / 2, grid_width / 2, spacing):
                start_y, finish_y = grid_height / 2, -grid_height / 2
                start_pt, finish_pt = world_to_plane(np.array([x, start_y, 0])), world_to_plane(np.array([x, finish_y, 0]))
                pygame.draw.line(window, white * 0.5, *A_many(shift_many([start_pt, finish_pt])), width=2)
            # Horizontal Lines
            for y in np.arange(-grid_height / 2, grid_height / 2, spacing):
                start_x, finish_x = -grid_width / 2, grid_width / 2
                start_pt, finish_pt = world_to_plane(np.array([start_x, y, 0])), world_to_plane(np.array([finish_x, y, 0]))
                pygame.draw.line(window, white * 0.5, *A_many(shift_many([start_pt, finish_pt])), width=2)
            # Origin
            radius = 13
            center2d = A(shift(world_to_plane(np.array([0, 0, 0]))))
            pygame.draw.circle(window, white, center2d, radius, width=0)

            # Draw traveler, and other markers for starting position.
            radius = 40
            center2d = world_to_plane(np.array([spacing * 5, -spacing * 5, 0]))
            blob_pts = get_traveler_pts(1, radius, *center2d, jiggle=0, du=0.001)
            pygame.draw.lines(window, white, False, A_many(shift_many(blob_pts)), width=5)
            pygame.draw.polygon(window, traveler_blue, A_many(shift_many(blob_pts)), 0)

            # Draw start text
            window.blit(text, A(shift(center2d + np.array([-100, -100]))))

        # Keyframe 2, 3, 4, 5 —— Traveler's journey + zoom out
        if frame in {2, 3, 4, 5}:
            if frame == 3:
                trace_line_col = np.array([116., 247., 64.])
            elif frame == 4:
                trace_line_col = np.array([32., 124., 245.])
            else:
                trace_line_col = np.array([255., 23., 89.])
            # Break the time apart
            tau, segment = squash(u, intervals=[0.0, 0.2, 0.25, 0.6, 0.65, 0.9, 1.0])
            spacing = 100.0
            frame_offset, frame_angle = (spacing * 5) * np.array([-1, 1]), 0
            traveler_xy = np.array([1, -1]) * spacing * 5

            if segment == 0:
                tau = ease_inout(tau)
                # Move the camera + the traveler
                if frame in {2, 3, 4}:
                    frame_offset += (tau * 10. * spacing * np.array([0., -1.]))
                else:
                    f0, f1 = 1000000., 1234.
                    sigma = slash(tau, 0.8)
                    sigma = ease_out(sigma)
                    tau = ease_out(tau)
                    focus = f0 * np.power(f1 / f0, min(1.0, 1.2 * tau))
                    frame_offset = (1 - sigma) * (spacing * 5) * np.array([-1, 1])
                    rho1, rho2 = 1000., 1200.
                    rho = ((1 - sigma) * rho1) + (sigma * rho2)

                traveler_xy += tau * np.array([0., 10. * spacing])
                shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
                # Trace out the line
                trace_start = A(shift(world_to_plane(np.array([spacing * 5, -spacing * 5, 0]))))
                trace_end = A(shift(world_to_plane(np.array([spacing * 5, spacing * 5, 0]))))
                trace_mid = ((1 - tau) * trace_start) + (tau * trace_end)
                pygame.draw.line(window, trace_line_col, trace_start, trace_mid, width=20)
                # Draw start text
                center2d = world_to_plane(np.array([spacing * 5, -spacing * 5, 0]))
                window.blit(text, A(shift(center2d + np.array([-100, -100]))))

            elif segment == 1:  # Pause + TODO: Rotate traveler
                # Keep the camera + traveler stationary
                if frame in {2, 3, 4}:
                    frame_offset += (10. * spacing * np.array([0., -1.]))
                else:
                    frame_offset = np.array([0,0])
                traveler_xy += np.array([0., 10. * spacing])
                shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
                # Keep the line as is.
                trace_start = A(shift(world_to_plane(np.array([spacing * 5, -spacing * 5, 0]))))
                trace_end = A(shift(world_to_plane(np.array([spacing * 5, spacing * 5, 0]))))
                pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
            elif segment == 2:
                tau = ease_inout(tau)
                # Move the camera + traveler
                if frame in {2, 3, 4}:
                    frame_offset = ((1 - tau) * 5 * spacing * np.array([-1, -1])) + (tau * spacing * 5 * np.array([1, -1]))
                else:
                    frame_offset = np.array([0,0])
                traveler_xy = (1 - tau) * 5 * spacing * np.array([1, 1]) + (tau * spacing * 5 * np.array([-1, 1]))
                shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
                # Keep the previous traceline (right edge of the square)
                trace_start = A(shift(world_to_plane(np.array([spacing * 5, -spacing * 5, 0]))))
                trace_end = A(shift(world_to_plane(np.array([spacing * 5, spacing * 5, 0]))))
                pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
                # Trace the new line (top edge of square)
                trace_start = copy.copy(trace_end)
                trace_end = A(shift(world_to_plane(np.array([-spacing * 5, spacing * 5, 0]))))
                trace_mid = ((1 - tau) * trace_start) + (tau * trace_end)
                pygame.draw.line(window, trace_line_col, trace_start, trace_mid, width=20)
            elif segment == 3:  # Pause + TODO: Rotate traveler
                # Keep camera + traveler stationary
                if frame in {2, 3, 4}:
                    frame_offset = spacing * 5 * np.array([1, -1])
                else:
                    frame_offset = np.array([0,0])
                traveler_xy = spacing * 5 * np.array([-1, 1])
                shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
                # Keep the first traceline, if we're zoomed out. Otherwise no need.
                if frame == 5:
                    trace_start = A(shift(world_to_plane(np.array([spacing * 5, -spacing * 5, 0]))))
                    trace_end = A(shift(world_to_plane(np.array([spacing * 5, spacing * 5, 0]))))
                    pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
                # Keep the one traceline (top edge of square) still in view
                trace_start = A(shift(world_to_plane(np.array([spacing * 5, spacing * 5, 0]))))
                trace_end = A(shift(world_to_plane(np.array([-spacing * 5, spacing * 5, 0]))))
                pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
            elif segment == 4:
                tau = ease_inout(tau)
                # Move the camera + traveler to the final stop (the beginning again)
                if frame in {2, 3, 4}:
                    frame_offset = ((1 - tau) * spacing * 5 * np.array([1, -1])) + (tau * spacing * 5 * np.array([1, 1]))
                else:
                    frame_offset = np.array([0,0])
                traveler_xy = (1 - tau) * 5 * spacing * np.array([-1, 1]) + (tau * spacing * 5 * np.array([-1, -1]))
                shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
                # Draw the first traceline (right square edge) if we're zoomed out. Otherwise no need.
                if frame == 5:
                    trace_start = A(shift(world_to_plane(np.array([spacing * 5, -spacing * 5, 0]))))
                    trace_end = A(shift(world_to_plane(np.array([spacing * 5, spacing * 5, 0]))))
                    pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
                # Keep the previous (top square edge) traceline
                trace_start = A(shift(world_to_plane(np.array([spacing * 5, spacing * 5, 0]))))
                trace_end = A(shift(world_to_plane(np.array([-spacing * 5, spacing * 5, 0]))))
                pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
                # Draw out the new traceline
                trace_start = copy.copy(trace_end)
                trace_end = A(shift(world_to_plane(np.array([-spacing * 5, -spacing * 5, 0]))))
                trace_mid = ((1 - tau) * trace_start) + (tau * trace_end)
                pygame.draw.line(window, trace_line_col, trace_start, trace_mid, width=20)
                if frame != 5:
                    # Show the "original" traceline which is in view again
                    trace_start = copy.copy(trace_end)
                    trace_end = A(shift(world_to_plane(np.array([spacing * 5, -spacing * 5, 0]))))
                    pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
                    # Show the "original" rotated text again. TODO: Make the position / rotation accurate
                    topleft2d = world_to_plane(np.array([-spacing * 5, -spacing * 5, 0])) + np.array([-100, 100])
                    rotated_text = pygame.transform.rotate(text, -90)
                    window.blit(rotated_text, A(shift(topleft2d)))
                    # rotated_rect = rotated_text.get_rect(center=rotated_text.get_rect().center)
            elif segment == 5:
                tau = ease_inout(tau)
                # Keep the frame location the same, but rotate the angle. Keep traveler as is. TODO: Rotate him!
                if frame in {2, 3, 4}:
                    frame_offset = spacing * 5 * np.array([1, 1])
                    frame_angle = tau * np.pi/2
                else:
                    frame_offset = np.array([0,0])
                    frame_angle = 0
                traveler_xy = spacing * 5 * np.array([-1, -1])
                if frame == 5:
                    # Keep the first traceline (right edge of the square)
                    shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
                    trace_start = A(shift(world_to_plane(np.array([spacing * 5, -spacing * 5, 0]))))
                    trace_end = A(shift(world_to_plane(np.array([spacing * 5, spacing * 5, 0]))))
                    pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
                    # Keep the second traceline (top edge of square) still in view
                    trace_start = A(shift(world_to_plane(np.array([spacing * 5, spacing * 5, 0]))))
                    trace_end = A(shift(world_to_plane(np.array([-spacing * 5, spacing * 5, 0]))))
                    pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
                    # Keep the third traceline (left edge of square) still in view
                    trace_start = copy.copy(trace_end)
                    trace_end = A(shift(world_to_plane(np.array([-spacing * 5, -spacing * 5, 0]))))
                    pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
                else:
                    # Keep the 2 tracelines still in view
                    shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
                    trace_start = A(shift(world_to_plane(np.array([-spacing * 5, spacing * 5, 0]))))
                    trace_end = A(shift(world_to_plane(np.array([-spacing * 5, -spacing * 5, 0]))))
                    pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
                    trace_start = copy.copy(trace_end)
                    trace_end = A(shift(world_to_plane(np.array([spacing * 5, -spacing * 5, 0]))))
                    pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
                    # Draw rotated text
                    topleft2d = world_to_plane(np.array([-spacing * 5, -spacing * 5, 0])) + np.array([-100, 100])
                    rotated_text = pygame.transform.rotate(text, -90 * (1 - tau))
                    window.blit(rotated_text, A(shift(topleft2d)))

            # Actually compute the frame shift
            shift_many = partial(frame_shift2d_many, offset=frame_offset, radians=frame_angle)
            shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
            # Vertical Lines
            grid_width, grid_height = width * 5, height * 5
            for x in np.arange(-grid_width / 2, grid_width / 2, spacing):
                start_y, finish_y = grid_height / 2, -grid_height / 2
                start_pt, finish_pt = world_to_plane(np.array([x, start_y, 0])), world_to_plane(
                    np.array([x, finish_y, 0]))
                pygame.draw.line(window, white * 0.5, *A_many(shift_many([start_pt, finish_pt])), width=2)
            # Horizontal Lines
            for y in np.arange(-grid_height / 2, grid_height / 2, spacing):
                start_x, finish_x = -grid_width / 2, grid_width / 2
                start_pt, finish_pt = world_to_plane(np.array([start_x, y, 0])), world_to_plane(
                    np.array([finish_x, y, 0]))
                pygame.draw.line(window, white * 0.5, *A_many(shift_many([start_pt, finish_pt])), width=2)

            # Draw traveler, and other markers for starting position.
            radius = 40
            if frame == 5 and segment == 0:
                radius = ((1 - tau) * 40) + (tau * 20)
            elif frame == 5:
                radius = 20
            center2d = world_to_plane(np.array([*traveler_xy, 0]))
            blob_pts = get_traveler_pts(1, radius, *center2d, jiggle=0, du=0.001)
            pygame.draw.lines(window, white, False, A_many(shift_many(blob_pts)), width=5)
            pygame.draw.polygon(window, traveler_blue, A_many(shift_many(blob_pts)), 0)

            # Origin
            if frame == 5:
                radius = 10
                center2d = A(shift(world_to_plane(np.array([0, 0, 0]))))
                pygame.draw.circle(window, white, center2d, radius, width=0)

        # Keyframe 6 —— Undo traveler's journey
        if frame == 6:
            u = ease_inout(slash(u, 0.8))
            tau, segment = squash2(u, 3)  # 1 interval for undo-ing each part of the journey
            # Recompute shift
            frame_offset = np.array([0, 0])
            frame_angle = 0
            shift_many = partial(frame_shift2d_many, offset=frame_offset, radians=frame_angle)
            # Draw the trace lines
            # First traceline (right edge of square)
            spacing = 100.0
            shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
            trace_start = A(shift(world_to_plane(np.array([spacing * 5, -spacing * 5, 0]))))
            y = ((1 - tau) * (spacing * 5)) + (tau * -spacing * 5) if segment == 2 else spacing * 5
            trace_end = A(shift(world_to_plane(np.array([spacing * 5, y, 0]))))
            pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
            # Second traceline (top edge of square)
            trace_start = A(shift(world_to_plane(np.array([spacing * 5, spacing * 5, 0]))))
            x = ((1 - tau) * (-spacing * 5)) + (tau * spacing * 5) if segment == 1 else -spacing * 5 if segment == 0 else spacing * 5
            trace_end = A(shift(world_to_plane(np.array([x, spacing * 5, 0]))))
            pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
            # Third traceline (left edge of square)
            trace_start = A(shift(world_to_plane(np.array([-spacing * 5, spacing * 5, 0]))))
            y = ((1 - tau) * (-spacing * 5)) + (tau * spacing * 5) if segment == 0 else spacing * 5
            trace_end = A(shift(world_to_plane(np.array([-spacing * 5, y, 0]))))
            pygame.draw.line(window, trace_line_col, trace_start, trace_end, width=20)
            # Redraw gridlines
            grid_width, grid_height = width * 5, height * 5
            for x in np.arange(-grid_width / 2, grid_width / 2, spacing):
                start_y, finish_y = grid_height / 2, -grid_height / 2
                start_pt, finish_pt = world_to_plane(np.array([x, start_y, 0])), world_to_plane(
                    np.array([x, finish_y, 0]))
                pygame.draw.line(window, white * 0.5, *A_many(shift_many([start_pt, finish_pt])), width=2)
            for y in np.arange(-grid_height / 2, grid_height / 2, spacing):
                start_x, finish_x = -grid_width / 2, grid_width / 2
                start_pt, finish_pt = world_to_plane(np.array([start_x, y, 0])), world_to_plane(
                    np.array([finish_x, y, 0]))
                pygame.draw.line(window, white * 0.5, *A_many(shift_many([start_pt, finish_pt])), width=2)
            # Redraw origin
            radius = 10
            center2d = A(shift(world_to_plane(np.array([0, 0, 0]))))
            pygame.draw.circle(window, white, center2d, radius, width=0)
            # Redraw traveler
            corners = [np.array([-1, -1]), np.array([-1, 1]), np.array([1, 1]), np.array([1, -1])]
            traveler_xy = (((1 - tau) * corners[segment]) + (tau * corners[segment + 1])) * spacing * 5
            radius = 20
            center2d = world_to_plane(np.array([*traveler_xy, 0]))
            blob_pts = get_traveler_pts(1, radius, *center2d, jiggle=0, du=0.001)
            pygame.draw.lines(window, white, False, A_many(shift_many(blob_pts)), width=5)
            pygame.draw.polygon(window, traveler_blue, A_many(shift_many(blob_pts)), 0)

        # Keyframe 7 —— Fade out portion of the plane and draw portals.
        if frame == 7:
            # Recompute shift
            frame_offset = np.array([0, 0])
            frame_angle = 0
            shift_many = partial(frame_shift2d_many, offset=frame_offset, radians=frame_angle)
            shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
            # Redraw gridlines
            spacing = 100.0
            grid_width, grid_height = width * 5, height * 5
            for x in np.arange(-grid_width / 2, grid_width / 2, spacing):
                start_y, finish_y = grid_height / 2, -grid_height / 2
                start_pt, finish_pt = world_to_plane(np.array([x, start_y, 0])), world_to_plane(
                    np.array([x, finish_y, 0]))
                pygame.draw.line(window, white * 0.5, *A_many(shift_many([start_pt, finish_pt])), width=2)
            for y in np.arange(-grid_height / 2, grid_height / 2, spacing):
                start_x, finish_x = -grid_width / 2, grid_width / 2
                start_pt, finish_pt = world_to_plane(np.array([start_x, y, 0])), world_to_plane(
                    np.array([finish_x, y, 0]))
                pygame.draw.line(window, white * 0.5, *A_many(shift_many([start_pt, finish_pt])), width=2)
            # Redraw traveler
            traveler_xy = np.array([1, -1]) * spacing * 5
            radius = 20
            center2d = world_to_plane(np.array([*traveler_xy, 0]))
            blob_pts = get_traveler_pts(1, radius, *center2d, jiggle=0, du=0.001)
            pygame.draw.lines(window, white, False, A_many(shift_many(blob_pts)), width=5)
            pygame.draw.polygon(window, traveler_blue, A_many(shift_many(blob_pts)), 0)

            # Fade out third quadrant of plane + draw portals
            blank = pygame.Surface((width/2, height/2))  # the size of your rect
            blank.set_alpha(255 * u)  # alpha level
            blank.fill(black)  # this fills the entire surface
            window.blit(blank, (0, height/2))  # (0,0) are the top-left coordinates

            # Draw portals
            pygame.draw.lines(window, np.array([76, 0, 255]), False,
                              [np.array([0, height/2]), np.array([width/2, height/2]),
                               np.array([width/2, height])], width=10)

            # Redraw origin
            radius = 10
            center2d = A(shift(world_to_plane(np.array([0, 0, 0]))))
            pygame.draw.circle(window, white, center2d, radius, width=0)

        # Keyframe 8 —— Show new path going through the portals.
        if frame == 8:
            # Recompute shift
            frame_offset = np.array([0, 0])
            frame_angle = 0
            shift_many = partial(frame_shift2d_many, offset=frame_offset, radians=frame_angle)
            shift = partial(frame_shift2d, offset=frame_offset, radians=frame_angle)
            # Redraw gridlines
            spacing = 100.0
            grid_width, grid_height = width * 5, height * 5
            for x in np.arange(-grid_width / 2, grid_width / 2, spacing):
                start_y, finish_y = grid_height / 2, -grid_height / 2
                start_pt, finish_pt = world_to_plane(np.array([x, start_y, 0])), world_to_plane(
                    np.array([x, finish_y, 0]))
                pygame.draw.line(window, white * 0.5, *A_many(shift_many([start_pt, finish_pt])), width=2)
            for y in np.arange(-grid_height / 2, grid_height / 2, spacing):
                start_x, finish_x = -grid_width / 2, grid_width / 2
                start_pt, finish_pt = world_to_plane(np.array([start_x, y, 0])), world_to_plane(
                    np.array([finish_x, y, 0]))
                pygame.draw.line(window, white * 0.5, *A_many(shift_many([start_pt, finish_pt])), width=2)
            # Redraw traveler
            traveler_xy = np.array([1, -1]) * spacing * 5
            radius = 20
            center2d = world_to_plane(np.array([*traveler_xy, 0]))
            blob_pts = get_traveler_pts(1, radius, *center2d, jiggle=0, du=0.001)
            pygame.draw.lines(window, white, False, A_many(shift_many(blob_pts)), width=5)
            pygame.draw.polygon(window, traveler_blue, A_many(shift_many(blob_pts)), 0)
            # Redraw cover of third quadrant
            blank = pygame.Surface((width/2, height/2))  # the size of your rect
            blank.set_alpha(255)  # alpha level
            blank.fill(black)  # this fills the entire surface
            window.blit(blank, (0, height/2))  # (0,0) are the top-left coordinates
            # Redraw portals
            pygame.draw.lines(window, np.array([76, 0, 255]), False,
                              [np.array([0, height/2]), np.array([width/2, height/2]),
                               np.array([width/2, height])], width=10)
            # Redraw origin
            radius = 10
            center2d = A(shift(world_to_plane(np.array([0, 0, 0]))))
            pygame.draw.circle(window, white, center2d, radius, width=0)

            # Demo path going through sectorial portal
            u = ease_inout(u)
            # pts = get_many_line_pts(u, np.array([-200, 200]), np.array([-200, -300]), du=0.01)
            # pts = get_circle_pts(u, 100, 0, 0, du=0.001)
            # pts = get_cosine_pts(u, -width/2, 0, scale=20, cycles=6, du=0.001)
            pts = get_many_line_pts(u, np.array([200, -500, 0]), np.array([-500, 300, 0]), du=0.01)
            radius /= 2
            pts, cut_indices = through_portals(pts)
            for k in range(len(cut_indices) - 1):
                segment = pts[cut_indices[k]:cut_indices[k+1]]
                if len(segment) > 2:
                    pygame.draw.lines(window, trace_line_col, False, A_many(world_to_plane_many(segment)), width=5)

        # Keyframe 9 —— 2D to cone transformation (DEMO)
        if frame == 9:
            u = slash(u, 0.2)
            # Redraw gridlines
            spacing = 100.0
            shift_many = partial(frame_shift2d_many, offset=np.array([0,0]), radians=0)
            grid_width, grid_height = width * 5, height * 5
            for x in np.arange(-grid_width / 2, grid_width / 2, spacing):
                start_y, finish_y = grid_height / 2, -grid_height / 2
                start_pt, finish_pt = world_to_plane(np.array([x, start_y, 0])), world_to_plane(
                    np.array([x, finish_y, 0]))
                pygame.draw.line(window, white * 0.1, *A_many(shift_many([start_pt, finish_pt])), width=2)
            for y in np.arange(-grid_height / 2, grid_height / 2, spacing):
                start_x, finish_x = -grid_width / 2, grid_width / 2
                start_pt, finish_pt = world_to_plane(np.array([start_x, y, 0])), world_to_plane(
                    np.array([finish_x, y, 0]))
                pygame.draw.line(window, white * 0.1, *A_many(shift_many([start_pt, finish_pt])), width=2)
            u = ease_inout(u)
            if u < 1.0:  # just for freedom to move around using keyboard
                phi0, phi1 = 0, np.pi/2.5
                phi = ((1 - u) * phi0) + (u * phi1)
                rho1, rho2 = 1200., 2000.
                rho = ((1 - u) * rho1) + (u * rho2)
            # Draw sample points (the ones which will actually move around to display the transformation)
            radius = 3
            n = 10
            for x in np.arange(-spacing * n, spacing * n, spacing):
                for y in np.arange(-spacing * n, spacing * n, spacing):
                    if x >= 0 or y >= 0:
                        domain_point0 = A(world_to_plane(to_cone(u, np.array([x, y, 0]))))
                        domain_point1 = A(world_to_plane(to_cone(u, np.array([x + spacing, y, 0]))))
                        domain_point2 = A(world_to_plane(to_cone(u, np.array([x, y + spacing, 0]))))
                        # if 0 <= domain_point0[0] <= width and 0 <= domain_point0[1] <= height:
                        #     pygame.draw.circle(window, white, domain_point0, radius, width=0)

                        pygame.draw.line(window, white, domain_point0, domain_point1, width=1)
                        pygame.draw.line(window, white, domain_point0, domain_point2, width=1)

            # Draw traveler location (in 3D)
            traveler_xy = np.array([1, -1]) * spacing * 5
            radius = 20
            center2d = world_to_plane(to_cone(u, np.array([*traveler_xy, 0])))
            blob_pts = get_traveler_pts(1, radius, *center2d, jiggle=0, du=0.001)
            pygame.draw.lines(window, white, False, A_many(shift_many(blob_pts)), width=5)
            pygame.draw.polygon(window, traveler_blue, A_many(shift_many(blob_pts)), 0)

            # Draw demo path location (in 3D)
            # pts = get_many_line_pts(1, np.array([-200, 200]), np.array([-200, -300]), du=0.01)
            # pts = get_circle_pts(1, 100, 0, 0, du=0.001)
            # pts = get_cosine_pts(1, -width/2, 0, scale=20, cycles=6, du=0.001)
            pts = get_many_line_pts(1, np.array([200, -500, 0]), np.array([-500, 300, 0]), du=0.01)
            radius /= 2
            pts, cut_indices = through_portals(pts)
            pts = to_cone_many(u, pts)
            for k in range(len(cut_indices) - 1):
                segment = pts[cut_indices[k]:cut_indices[k+1]]
                if len(segment) > 2:
                    pygame.draw.lines(window, trace_line_col, False, A_many(world_to_plane_many(segment)), width=5)


        # We handle keys pressed inside the gameloop in PyGame
        keys_pressed = pygame.key.get_pressed()
        handle_keys(keys_pressed)


        # End run (1. Tick clock, 2. Save the frame, and 3. Detect if window closed)
        pygame.display.update()
        t += dt
        clock.tick(FPS)
        count += 1
        if save_anim:
            pygame.image.save(window, path_to_save+'/frame'+str(count)+'.png')
            print('Saved frame '+str(count))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()


    # Post game-loop stuff
    # Do more stuff...
    # Use ffmpeg to combine the PNG images into a video
    if save_anim:
        input_files = path_to_save + '/frame%d.png'
        output_file = path_to_save + '/output.mp4'
        ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
        os.system(f'{ffmpeg_path} -r 60 -i {input_files} -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "eq=brightness=0.00:saturation=1.3" {output_file} > /dev/null 2>&1')
        print('Saved video to ' + output_file)


if __name__ == "__main__":
    main()
