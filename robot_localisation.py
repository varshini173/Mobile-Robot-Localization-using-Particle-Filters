import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# ==========================================================
# FINAL ADVANCED MOBILE ROBOT LOCALIZATION
# PARTICLE FILTER + LANDMARK LOCALIZATION
# ==========================================================

# -----------------------------
# CONFIGURATION
# -----------------------------

GRID_SIZE = 6
NUM_PARTICLES = 1200
STEPS = 12

# -----------------------------
# ENVIRONMENT CODES
# -----------------------------

EMPTY = 0
WALL = 1
BEACON = 2
CHARGING = 3

# -----------------------------
# CREATE ENVIRONMENT
# -----------------------------

world = np.array([

    [0,0,0,2,0,0],
    [0,1,0,0,3,0],
    [0,0,1,0,0,0],
    [2,0,0,0,1,0],
    [0,3,0,0,0,0],
    [0,0,0,1,0,2]

])

# -----------------------------
# COLORS FOR DISPLAY
# -----------------------------

environment_colors = {
    WALL: 'black',
    BEACON: 'purple',
    CHARGING: 'orange'
}

# -----------------------------
# INITIALIZE ROBOT
# -----------------------------

while True:

    robot_x = random.randint(0, GRID_SIZE - 1)
    robot_y = random.randint(0, GRID_SIZE - 1)

    if world[robot_x, robot_y] != WALL:
        break

# -----------------------------
# INITIALIZE PARTICLES
# -----------------------------

particles = []

for _ in range(NUM_PARTICLES):

    while True:

        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)

        if world[x, y] != WALL:
            particles.append([x, y])
            break

particles = np.array(particles)

weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

# ==========================================================
# SENSOR MODEL
# ==========================================================

def get_sensor_signature(x, y):

    # Nearby directional walls
    directions = [
        (-1,0),
        (1,0),
        (0,-1),
        (0,1)
    ]

    wall_signature = []

    for dx, dy in directions:

        nx = x + dx
        ny = y + dy

        if (
            nx < 0 or
            nx >= GRID_SIZE or
            ny < 0 or
            ny >= GRID_SIZE
        ):
            wall_signature.append(1)

        elif world[nx, ny] == WALL:
            wall_signature.append(1)

        else:
            wall_signature.append(0)

    # Landmark detection
    landmark = world[x, y]

    return wall_signature, landmark

# ==========================================================
# ROBOT MOVEMENT
# ==========================================================

last_move = (0,0)

def move_robot():

    global robot_x, robot_y, last_move

    moves = [
        (-1,0),
        (1,0),
        (0,-1),
        (0,1)
    ]

    dx, dy = random.choice(moves)

    new_x = robot_x + dx
    new_y = robot_y + dy

    if (
        0 <= new_x < GRID_SIZE and
        0 <= new_y < GRID_SIZE and
        world[new_x, new_y] != WALL
    ):

        robot_x = new_x
        robot_y = new_y

        last_move = (dx, dy)

# ==========================================================
# PARTICLE MOVEMENT
# ==========================================================

def move_particles():

    global particles

    dx, dy = last_move

    for i in range(NUM_PARTICLES):

        # Most particles follow robot direction
        if random.random() < 0.92:

            # Reduced noise
            noise_x = random.choice([0,0,0,-1,1])
            noise_y = random.choice([0,0,0,-1,1])

            new_x = particles[i][0] + dx + noise_x
            new_y = particles[i][1] + dy + noise_y

        else:

            # Small random error
            new_x = particles[i][0] + random.choice([-1,0,1])
            new_y = particles[i][1] + random.choice([-1,0,1])

        if (
            0 <= new_x < GRID_SIZE and
            0 <= new_y < GRID_SIZE and
            world[new_x, new_y] != WALL
        ):

            particles[i][0] = new_x
            particles[i][1] = new_y

# ==========================================================
# WEIGHT UPDATE
# ==========================================================

def update_weights():

    global weights

    robot_walls, robot_landmark = get_sensor_signature(
        robot_x,
        robot_y
    )

    for i in range(NUM_PARTICLES):

        px, py = particles[i]

        particle_walls, particle_landmark = get_sensor_signature(
            px,
            py
        )

        # Wall similarity
        matches = 0

        for r, p in zip(robot_walls, particle_walls):

            if r == p:
                matches += 1

        wall_similarity = matches / 4.0

        # Landmark bonus
        landmark_bonus = 0

        if robot_landmark == particle_landmark:
            landmark_bonus = 1.5

        # Final weight
        weights[i] = wall_similarity + landmark_bonus + 0.001

    # Normalize
    weights /= np.sum(weights)

# ==========================================================
# RESAMPLING
# ==========================================================

def resample_particles():

    global particles, weights

    indices = np.random.choice(

        range(NUM_PARTICLES),
        size=NUM_PARTICLES,
        p=weights

    )

    particles = particles[indices]

    weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

# ==========================================================
# WEIGHTED POSITION ESTIMATION
# ==========================================================

def estimate_position():

    x_est = np.average(
        particles[:,0],
        weights=weights
    )

    y_est = np.average(
        particles[:,1],
        weights=weights
    )

    return int(round(x_est)), int(round(y_est))

# ==========================================================
# CREATE HEATMAP
# ==========================================================

def create_heatmap():

    heatmap = np.zeros((GRID_SIZE, GRID_SIZE))

    for particle in particles:

        x, y = particle
        heatmap[x, y] += 1

    heatmap /= np.max(heatmap)

    return heatmap

# ==========================================================
# VISUALIZATION
# ==========================================================

fig, ax = plt.subplots(figsize=(8,8))

# ==========================================================
# ANIMATION FUNCTION
# ==========================================================

def animate(frame):

    ax.clear()

    # -----------------------------
    # UPDATE LOCALIZATION
    # -----------------------------

    move_robot()

    move_particles()

    update_weights()

    resample_particles()

    est_x, est_y = estimate_position()

    heatmap = create_heatmap()

    # -----------------------------
    # DRAW HEATMAP
    # -----------------------------

    ax.imshow(

        heatmap,
        cmap='Blues',
        origin='lower',
        alpha=0.7

    )

    # -----------------------------
    # DRAW ENVIRONMENT
    # -----------------------------

    for x in range(GRID_SIZE):

        for y in range(GRID_SIZE):

            cell = world[x, y]

            if cell in environment_colors:

                ax.add_patch(

                    plt.Rectangle(

                        (y - 0.5, x - 0.5),
                        1,
                        1,
                        color=environment_colors[cell]

                    )

                )

    # -----------------------------
    # SHOW FEW PARTICLES
    # -----------------------------

    sample_indices = np.random.choice(

        range(NUM_PARTICLES),
        120,
        replace=False

    )

    sample_particles = particles[sample_indices]

    ax.scatter(

        sample_particles[:,1],
        sample_particles[:,0],

        s=10,
        alpha=0.08,
        label='Particles'

    )

    # -----------------------------
    # ACTUAL ROBOT
    # -----------------------------

    ax.scatter(

        robot_y,
        robot_x,

        s=280,
        marker='o',
        color='red',

        label='Actual Robot'

    )

    # -----------------------------
    # ESTIMATED POSITION
    # -----------------------------

    ax.scatter(

        est_y,
        est_x,

        s=320,
        marker='X',
        color='lime',

        label='Estimated Position'

    )

    # -----------------------------
    # FORMATTING
    # -----------------------------

    ax.set_title(

        f'Advanced Mobile Robot Localization | Step {frame+1}'

    )

    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))

    ax.grid(True)

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)

    ax.legend(loc='upper right')

    # -----------------------------
    # CONSOLE OUTPUT
    # -----------------------------

    print(f'\nSTEP {frame+1}')

    print(
        f'Actual Position: ({robot_x}, {robot_y})'
    )

    print(
        f'Estimated Position: ({est_x}, {est_y})'
    )

    walls, landmark = get_sensor_signature(
        robot_x,
        robot_y
    )

    print(
        f'Wall Sensor: {walls}'
    )

    print(
        f'Landmark Detected: {landmark}'
    )

# ==========================================================
# RUN ANIMATION
# ==========================================================

ani = animation.FuncAnimation(

    fig,
    animate,

    frames=STEPS,
    interval=1200,

    repeat=False

)

plt.show()