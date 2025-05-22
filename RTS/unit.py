import asyncio
import platform
import heapq
import numpy as np
import pygame
from typing import List, Tuple, Optional

# Unit class (unchanged)
class Unit:
    def __init__(self, pos: Tuple[float, float], goal: Tuple[float, float], speed: float = 2.5, team: str = "ally"):
        self.pos = np.array(pos, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.path = []
        self.speed = speed
        self.selected = False
        self.velocity = np.zeros(2)
        self.forces = {'separation': np.zeros(2), 'cohesion': np.zeros(2), 
                      'alignment': np.zeros(2), 'obstacle': np.zeros(2), 'influence': np.zeros(2)}
        self.angle = 0
        self.team = team
        self.last_goal = np.array(goal, dtype=float)  # Track last goal for path caching

# Manhattan distance (unchanged)
def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Jump Point Search (optimized with caching check)
def jump_point_search(start: Tuple[int, int], goal: Tuple[int, int], grid: np.ndarray) -> List[Tuple[int, int]]:
    open_list = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}
    
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
    costs = [1.0, 1.0, 1.0, 1.0, 1.4, 1.4, 1.4, 1.4]

    def is_jump_point(current, direction, grid):
        x, y = current
        dx, dy = direction
        nx, ny = x + dx, y + dy
        if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0):
            return False
        if dx != 0 and dy != 0:
            if (0 <= x < grid.shape[0] and 0 <= y - dy < grid.shape[1] and grid[x, y - dy] == 1 and
                0 <= x + dx < grid.shape[0] and grid[x + dx, y - dy] == 0):
                return True
            if (0 <= x - dx < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x - dx, y] == 1 and
                0 <= x - dx < grid.shape[0] and 0 <= y + dy < grid.shape[1] and grid[x - dx, y + dy] == 0):
                return True
        return False

    while open_list:
        current = heapq.heappop(open_list)[1]
        if manhattan_distance(current, goal) < 1:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for (dx, dy), cost in zip(directions, costs):
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < grid.shape[0] and 
                0 <= neighbor[1] < grid.shape[1] and 
                grid[neighbor] == 0):
                jump_point = neighbor
                jump_dist = 1
                while True:
                    next_jump = (jump_point[0] + dx, jump_point[1] + dy)
                    if not (0 <= next_jump[0] < grid.shape[0] and 0 <= next_jump[1] < grid.shape[1] and grid[next_jump] == 0):
                        break
                    if is_jump_point(next_jump, (dx, dy), grid) or next_jump == goal:
                        jump_point = next_jump
                        jump_dist += 1
                    else:
                        break
                tentative_g = g_score[current] + cost * jump_dist
                if tentative_g < g_score.get(jump_point, float('inf')):
                    came_from[jump_point] = current
                    g_score[jump_point] = tentative_g
                    f_score[jump_point] = tentative_g + manhattan_distance(jump_point, goal)
                    heapq.heappush(open_list, (f_score[jump_point], jump_point))
    return []

# Hierarchical pathfinding (optimized with caching)
def hierarchical_pathfinding(start: Tuple[int, int], goal: Tuple[int, int], fine_grid: np.ndarray, coarse_grid: np.ndarray) -> List[Tuple[int, int]]:
    coarse_start = (start[0] // 4, start[1] // 4)
    coarse_goal = (goal[0] // 4, goal[1] // 4)
    coarse_path = jump_point_search(coarse_start, coarse_goal, coarse_grid)
    
    if not coarse_path:
        return jump_point_search(start, goal, fine_grid)
    
    path = []
    for i in range(len(coarse_path) - 1):
        segment_start = (coarse_path[i][0] * 4, coarse_path[i][1] * 4)
        segment_goal = (coarse_path[i + 1][0] * 4, coarse_path[i + 1][1] * 4)
        segment_path = jump_point_search(segment_start, segment_goal, fine_grid)
        path.extend(segment_path[:-1])
    path.append((goal[0], goal[1]))
    return path

# Spatial partitioning for flocking
def get_neighbors(unit: Unit, units: List[Unit], cell_size: float = 20.0) -> List[Unit]:
    cell_x, cell_y = int(unit.pos[0] // cell_size), int(unit.pos[1] // cell_size)
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for other in units:
                if other is unit:
                    continue
                other_cell_x, other_cell_y = int(other.pos[0] // cell_size), int(other.pos[1] // cell_size)
                if other_cell_x == cell_x + dx and other_cell_y == cell_y + dy:
                    neighbors.append(other)
    return neighbors

# Influence map (optimized with caching)
def create_influence_map(obstacles: List[Tuple[int, int]], units: List[Unit], grid_shape: Tuple[int, int]) -> np.ndarray:
    influence = np.zeros(grid_shape, dtype=float)
    for i, j in obstacles:
        influence[i, j] = 10.0
    for unit in units:
        if unit.team == "enemy":
            i, j = int(unit.pos[0]), int(unit.pos[1])
            influence[i, j] = 5.0
    return influence

# Flocking algorithm (optimized with spatial partitioning)
def apply_flocking(units: List[Unit], team: str, separation: float = 15.0, cohesion: float = 0.1, 
                  alignment: float = 0.05, obstacle_grid: np.ndarray = None, influence_map: np.ndarray = None) -> None:
    if not units:
        return
    
    for unit in units:
        if unit.team != team:
            continue
        unit.forces = {'separation': np.zeros(2), 'cohesion': np.zeros(2), 
                      'alignment': np.zeros(2), 'obstacle': np.zeros(2), 'influence': np.zeros(2)}
        count = 0

        # Use spatial partitioning to get nearby units
        neighbors = get_neighbors(unit, units)
        for other in neighbors:
            if other.team != team:
                continue
            diff = unit.pos - other.pos
            dist = np.linalg.norm(diff)
            if dist < separation and dist > 0:
                unit.forces['separation'] += diff / (dist * dist)
                count += 1
        
        if obstacle_grid is not None:
            grid_pos = (int(unit.pos[0]), int(unit.pos[1]))
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_pos = (grid_pos[0] + dx, grid_pos[1] + dy)
                    if (0 <= check_pos[0] < obstacle_grid.shape[0] and 
                        0 <= check_pos[1] < obstacle_grid.shape[1] and 
                        obstacle_grid[check_pos] == 1):
                        diff = unit.pos - np.array(check_pos, dtype=float)
                        dist = np.linalg.norm(diff)
                        if dist < separation and dist > 0:
                            unit.forces['obstacle'] += diff / (dist * dist)

        if influence_map is not None:
            grid_pos = (int(unit.pos[0]), int(unit.pos[1]))
            if (0 <= grid_pos[0] < influence_map.shape[0] and 
                0 <= grid_pos[1] < influence_map.shape[1] and 
                influence_map[grid_pos] > 0):
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        check_pos = (grid_pos[0] + dx, grid_pos[1] + dy)
                        if (0 <= check_pos[0] < influence_map.shape[0] and 
                            0 <= check_pos[1] < influence_map.shape[1]):
                            diff = unit.pos - np.array(check_pos, dtype=float)
                            dist = np.linalg.norm(diff)
                            if dist > 0:
                                unit.forces['influence'] -= diff / dist * influence_map[check_pos]

        if count > 0:
            unit.forces['separation'] /= count
        
        other_positions = [u.pos for u in neighbors if u.team == team]
        if other_positions:
            unit.forces['cohesion'] = np.mean(other_positions, axis=0) - unit.pos
        
        directions = [u.velocity for u in neighbors if u.team == team and np.any(u.velocity)]
        if directions:
            unit.forces['alignment'] = np.mean(directions, axis=0)
        
        total_force = (unit.forces['separation'] * 2.0 + 
                      unit.forces['cohesion'] * cohesion + 
                      unit.forces['alignment'] * alignment + 
                      unit.forces['obstacle'] * 2.0 + 
                      unit.forces['influence'] * 1.0)
        unit.velocity = unit.velocity * 0.9 + total_force * 0.1
        unit.pos += unit.velocity * unit.speed
        if np.linalg.norm(unit.velocity) > 0:
            unit.angle = np.arctan2(unit.velocity[1], unit.velocity[0]) * 180 / np.pi

# Create unit sprite (precompute rotations)
def create_unit_sprite(selected: bool = False, team: str = "ally") -> List[pygame.Surface]:
    surface = pygame.Surface((12, 12), pygame.SRCALPHA)
    color = (0, 255, 0) if selected else (0, 100, 255) if team == "ally" else (255, 0, 0)
    pygame.draw.circle(surface, color, (6, 6), 6)
    if selected:
        pygame.draw.circle(surface, (255, 255, 0), (6, 6), 6, 1)
    # Precompute rotations for common angles
    rotations = {}
    for angle in range(0, 360, 5):  # 5-degree increments
        rotations[angle] = pygame.transform.rotate(surface, -angle)
    return rotations

# Background texture (unchanged)
def create_background_texture() -> pygame.Surface:
    surface = pygame.Surface((800, 800))
    surface.fill((100, 150, 50))
    for _ in range(1000):
        x, y = np.random.randint(0, 800, 2)
        pygame.draw.circle(surface, (80, 120, 40), (x, y), np.random.randint(1, 3))
    return surface

# Obstacle texture (unchanged)
def create_obstacle_texture() -> pygame.Surface:
    surface = pygame.Surface((8, 8))
    surface.fill((0, 0, 0))
    return surface

# Wedge formation (unchanged)
def create_wedge_formation(start_pos: Tuple[float, float], num_units: int) -> List[Tuple[float, float]]:
    positions = []
    rows = int(np.ceil(np.sqrt(num_units)))
    for i in range(num_units):
        row = i // rows
        col = i % rows
        x = start_pos[0] + col - row / 2
        y = start_pos[1] + row
        positions.append((x, y))
    return positions

# Enemy movement (unchanged)
def update_enemy_movement(unit: Unit, grid: np.ndarray):
    if np.random.random() < 0.05:
        dx, dy = np.random.randint(-10, 11, 2)
        new_goal = (min(max(unit.goal[0] + dx, 0), grid.shape[0] - 1),
                    min(max(unit.goal[1] + dy, 0), grid.shape[1] - 1))
        if grid[int(new_goal[0]), int(new_goal[1])] == 0:
            unit.goal = np.array(new_goal, dtype=float)

# Main game loop
async def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    
    # Store obstacles as a list for sparse representation
    grid = np.zeros((100, 100))
    obstacles = []
    for i in range(20, 30):
        for j in range(20, 80):
            grid[i, j] = 1
            obstacles.append((i, j))
    for i in range(50, 80):
        for j in range(40, 50):
            grid[i, j] = 1
            obstacles.append((i, j))
    for i in range(60, 70):
        for j in range(10, 30):
            grid[i, j] = 1
            obstacles.append((i, j))
    
    coarse_grid = np.zeros((25, 25))
    for i in range(25):
        for j in range(25):
            if np.any(grid[i*4:(i+1)*4, j*4:(j+1)*4] == 1):
                coarse_grid[i, j] = 1
    
    num_allies = 50
    ally_positions = create_wedge_formation((10, 10), num_allies)
    units = [Unit(pos, (90, 90), speed=2.5, team="ally") for pos in ally_positions]
    
    num_enemies = 15
    enemy_positions = [(np.random.randint(20, 80), np.random.randint(20, 80)) for _ in range(num_enemies)]
    units.extend([Unit(pos, pos, speed=2.0, team="enemy") for pos in enemy_positions])
    
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    background = create_background_texture()
    obstacle_texture = create_obstacle_texture()
    ally_sprites = create_unit_sprite(team="ally")
    ally_selected_sprites = create_unit_sprite(selected=True, team="ally")
    enemy_sprites = create_unit_sprite(team="enemy")
    enemy_selected_sprites = create_unit_sprite(selected=True, team="enemy")
    
    selecting = False
    start_pos = (0, 0)
    current_pos = (0, 0)
    show_grid = True
    show_forces = False
    selected_unit = None
    time = 0
    FPS = 60
    influence_map = create_influence_map(obstacles, units, grid.shape)
    enemies_moved = False

    running = True
    while running:
        time += 1 / FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    selecting = True
                    start_pos = event.pos
                    current_pos = event.pos
                    for unit in units:
                        if unit.team == "ally" and np.linalg.norm(np.array([unit.pos[1] * 8, unit.pos[0] * 8]) - 
                                                                 np.array(event.pos)) < 6:
                            selected_unit = unit
                            break
                    else:
                        selected_unit = None
                elif event.button == 3:
                    new_goal = (event.pos[1] // 8, event.pos[0] // 8)
                    for unit in units:
                        if unit.selected and unit.team == "ally":
                            unit.goal = np.array(new_goal, dtype=float)
                            unit.last_goal = np.array(new_goal, dtype=float)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    selecting = False
                    rect = pygame.Rect(min(start_pos[0], current_pos[0]),
                                     min(start_pos[1], current_pos[1]),
                                     abs(start_pos[0] - current_pos[0]),
                                     abs(start_pos[1] - current_pos[1]))
                    for unit in units:
                        unit.selected = rect.collidepoint(unit.pos[1] * 8, unit.pos[0] * 8) and unit.team == "ally"
            elif event.type == pygame.MOUSEMOTION:
                if selecting:
                    current_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    show_grid = not show_grid
                elif event.key == pygame.K_f:
                    show_forces = not show_forces

        enemies_moved = False
        for unit in units:
            if unit.team == "enemy":
                old_goal = unit.goal.copy()
                update_enemy_movement(unit, grid)
                if not np.array_equal(old_goal, unit.goal):
                    enemies_moved = True

        if enemies_moved:
            influence_map = create_influence_map(obstacles, units, grid.shape)

        for unit in units:
            # Only recompute path if goal changed or path is empty
            if not unit.path or not np.array_equal(unit.goal, unit.last_goal) or np.linalg.norm(unit.pos - unit.goal) < 1:
                unit.path = hierarchical_pathfinding((int(unit.pos[0]), int(unit.pos[1])), 
                                                   (int(unit.goal[0]), int(unit.goal[1])), grid, coarse_grid)
                unit.last_goal = unit.goal.copy()
            if unit.path:
                next_pos = np.array(unit.path[0], dtype=float)
                direction = next_pos - unit.pos
                dist = np.linalg.norm(direction)
                if dist > 0:
                    unit.velocity = direction / dist * unit.speed
                if dist < 1:
                    unit.path.pop(0)

        apply_flocking(units, "ally", obstacle_grid=grid, influence_map=influence_map)
        apply_flocking(units, "enemy", obstacle_grid=grid, influence_map=influence_map)

        screen.blit(background, (0, 0))
        
        if show_grid:
            for i in range(0, 800, 8):
                pygame.draw.line(screen, (120, 120, 120), (i, 0), (i, 800), 1)
                pygame.draw.line(screen, (120, 120, 120), (0, i), (800, i), 1)
        
        for i, j in obstacles:
            screen.blit(obstacle_texture, (j * 8, i * 8))
        
        for unit in units:
            if unit.path:
                points = [(p[1] * 8 + 4, p[0] * 8 + 4) for p in unit.path]
                if len(points) > 1:
                    pygame.draw.lines(screen, (180, 180, 180), False, points, 1)
            
            scale = 4 + np.sin(time * 2) * 0.5
            pygame.draw.rect(screen, (255, 50, 50), 
                           (unit.goal[1] * 8 - scale/2, unit.goal[0] * 8 - scale/2, scale, scale))
            
            # Use precomputed rotated sprite
            angle = int(unit.angle // 5 * 5) % 360
            sprite_dict = ally_selected_sprites if unit.selected and unit.team == "ally" else \
                         enemy_selected_sprites if unit.selected and unit.team == "enemy" else \
                         ally_sprites if unit.team == "ally" else enemy_sprites
            sprite = sprite_dict[angle]
            rect = sprite.get_rect(center=(int(unit.pos[1] * 8), int(unit.pos[0] * 8)))
            screen.blit(sprite, rect)
            
            if show_forces and unit.selected:
                pos = np.array([unit.pos[1] * 8, unit.pos[0] * 8])
                scale = 20
                colors = {'separation': (255, 0, 0), 'cohesion': (0, 255, 0), 
                         'alignment': (0, 0, 255), 'obstacle': (255, 165, 0), 'influence': (255, 0, 255)}
                for force_name, force in unit.forces.items():
                    if np.linalg.norm(force) > 0:
                        end_pos = pos + force * scale
                        pygame.draw.line(screen, colors[force_name], pos, end_pos, 2)

        if selecting:
            rect = pygame.Rect(min(start_pos[0], current_pos[0]),
                             min(start_pos[1], current_pos[1]),
                             abs(start_pos[0] - current_pos[0]),
                             abs(start_pos[1] - current_pos[1]))
            pygame.draw.rect(screen, (0, 255, 0), rect, 1)
        
        pygame.draw.rect(screen, (230, 230, 230), (800, 0, 100, 800))
        status_texts = [
            f"FPS: {int(clock.get_fps())}",
            f"Units: {len(units)}",
            f"Grid: {'On' if show_grid else 'Off'} (G)",
            f"Forces: {'On' if show_forces else 'Off'} (F)",
        ]
        if selected_unit:
            status_texts.extend([
                f"Selected Unit ({selected_unit.team}):",
                f"Pos: ({selected_unit.pos[0]:.1f}, {selected_unit.pos[1]:.1f})",
                f"Goal: ({selected_unit.goal[0]:.1f}, {selected_unit.goal[1]:.1f})",
                f"Path Len: {len(selected_unit.path)}",
                f"Sep: {np.linalg.norm(selected_unit.forces['separation']):.2f}",
                f"Coh: {np.linalg.norm(selected_unit.forces['cohesion']):.2f}",
                f"Aln: {np.linalg.norm(selected_unit.forces['alignment']):.2f}",
                f"Obs: {np.linalg.norm(selected_unit.forces['obstacle']):.2f}",
                f"Inf: {np.linalg.norm(selected_unit.forces['influence']):.2f}"
            ])
        for i, text in enumerate(status_texts):
            screen.blit(font.render(text, True, (0, 0, 0)), (810, 10 + i * 30))
        
        pygame.display.flip()
        clock.tick(FPS)
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
