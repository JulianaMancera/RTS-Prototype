import heapq
import numpy as np
import pygame
from typing import List, Tuple

class Unit:
    def __init__(self, pos: Tuple[float, float], goal: Tuple[float, float], speed: float = 2.0):
        self.pos = np.array(pos, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.path = []
        self.speed = speed
        self.selected = False  # For unit selection
        self.velocity = np.zeros(2)  # For smoother movement

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def a_star(start: Tuple[int, int], goal: Tuple[int, int], grid: np.ndarray) -> List[Tuple[int, int]]:
    """Optimized A* pathfinding with early exit and diagonal movement"""
    open_list = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}
    
    # Add diagonal movements for smoother paths
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
    costs = [1.0, 1.0, 1.0, 1.0, 1.4, 1.4, 1.4, 1.4]  # Diagonal moves cost sqrt(2)

    while open_list:
        current = heapq.heappop(open_list)[1]
        if manhattan_distance(current, goal) < 1:  # Early exit condition
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
                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + manhattan_distance(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
    return []

def apply_flocking(units: List[Unit], separation: float = 15.0, cohesion: float = 0.1, 
                  alignment: float = 0.05, obstacle_grid: np.ndarray = None) -> None:
    """Enhanced flocking with obstacle avoidance"""
    if not units:
        return
    
    grid_scale = 8  # Pixel-to-grid scale factor
    for unit in units:
        separation_force = np.zeros(2)
        cohesion_force = np.zeros(2)
        alignment_force = np.zeros(2)
        obstacle_force = np.zeros(2)
        count = 0

        # Separation and obstacle avoidance
        for other in units:
            if other is unit:
                continue
            diff = unit.pos - other.pos
            dist = np.linalg.norm(diff)
            if dist < separation and dist > 0:
                separation_force += diff / (dist * dist)
                count += 1
        
        # Obstacle avoidance
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
                            obstacle_force += diff / (dist * dist)

        if count > 0:
            separation_force /= count
        
        # Cohesion and alignment
        other_positions = [u.pos for u in units if u is not unit]
        if other_positions:
            cohesion_force = np.mean(other_positions, axis=0) - unit.pos
        
        directions = [u.velocity for u in units if u is not unit and np.any(u.velocity)]
        if directions:
            alignment_force = np.mean(directions, axis=0)
        
        # Combine forces and update velocity
        total_force = (separation_force * 2.0 + 
                      cohesion_force * cohesion + 
                      alignment_force * alignment + 
                      obstacle_force * 2.0)
        unit.velocity = unit.velocity * 0.9 + total_force * 0.1  # Smooth velocity
        unit.pos += unit.velocity * unit.speed

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("RTS Pathfinding with Flocking")
    grid = np.zeros((100, 100))  # 0: free, 1: obstacle
    grid[20:30, 20:80] = 1  # Horizontal wall
    grid[50:80, 40:50] = 1  # Vertical wall
    
    units = [Unit((10, 10), (90, 90), speed=2.5) for _ in range(20)]  # Reduced units for performance
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Selection rectangle
    selecting = False
    start_pos = (0, 0)
    current_pos = (0, 0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click to start selection
                    selecting = True
                    start_pos = event.pos
                    current_pos = event.pos
                elif event.button == 3:  # Right click to set new goal
                    new_goal = (event.pos[1] // 8, event.pos[0] // 8)
                    for unit in units:
                        if unit.selected:
                            unit.goal = np.array(new_goal, dtype=float)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # End selection
                    selecting = False
                    # Select units within rectangle
                    rect = pygame.Rect(min(start_pos[0], current_pos[0]),
                                     min(start_pos[1], current_pos[1]),
                                     abs(start_pos[0] - current_pos[0]),
                                     abs(start_pos[1] - current_pos[1]))
                    for unit in units:
                        unit.selected = rect.collidepoint(unit.pos[1] * 8, unit.pos[0] * 8)
            elif event.type == pygame.MOUSEMOTION:
                if selecting:
                    current_pos = event.pos

        # Update units
        for unit in units:
            if not unit.path or np.linalg.norm(unit.pos - unit.goal) < 1:
                unit.path = a_star((int(unit.pos[0]), int(unit.pos[1])), 
                                 (int(unit.goal[0]), int(unit.goal[1])), grid)
            if unit.path:
                next_pos = np.array(unit.path[0], dtype=float)
                direction = next_pos - unit.pos
                dist = np.linalg.norm(direction)
                if dist > 0:
                    unit.velocity = direction / dist * unit.speed
                if dist < 1:
                    unit.path.pop(0)

        apply_flocking(units, obstacle_grid=grid)

        # Render
        screen.fill((255, 255, 255))
        # Draw grid
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 1:
                    pygame.draw.rect(screen, (50, 50, 50), (j * 8, i * 8, 8, 8))
        
        # Draw paths and units
        for unit in units:
            if unit.path:
                points = [(p[1] * 8 + 4, p[0] * 8 + 4) for p in unit.path]
                if len(points) > 1:
                    pygame.draw.lines(screen, (200, 200, 200), False, points, 1)
            color = (0, 255, 0) if unit.selected else (0, 0, 255)
            pygame.draw.circle(screen, color, (int(unit.pos[1] * 8), int(unit.pos[0] * 8)), 4)
            if unit.selected:
                pygame.draw.circle(screen, (255, 255, 0), 
                                 (int(unit.pos[1] * 8), int(unit.pos[0] * 8)), 6, 1)
        
        # Draw selection rectangle
        if selecting:
            rect = pygame.Rect(min(start_pos[0], current_pos[0]),
                             min(start_pos[1], current_pos[1]),
                             abs(start_pos[0] - current_pos[0]),
                             abs(start_pos[1] - current_pos[1]))
            pygame.draw.rect(screen, (0, 255, 0), rect, 1)
        
        # Draw FPS
        fps = str(int(clock.get_fps()))
        fps_text = font.render(fps, True, (0, 0, 0))
        screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()