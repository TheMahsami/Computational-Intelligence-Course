import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


# create maze
maze_data = np.array([
    # z=0
    [[0,1,1,1,1,1,1,1,1,1],
     [0,1,0,0,0,1,0,0,0,1],
     [0,0,0,0,0,0,0,1,0,1],
     [1,1,1,1,0,1,0,1,0,1],
     [0,0,0,0,0,1,0,1,0,1],
     [0,1,1,1,1,1,0,1,0,1],
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,1,1,1,1,1,1,1,0],
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,1,1,1,1,1,1,0,0]],

    # z=1
    [[1,1,1,1,1,1,1,1,1,1],
     [0,1,0,0,0,1,0,0,0,1],
     [0,1,0,1,0,1,0,1,0,1],
     [0,1,0,1,0,1,0,1,0,1],
     [0,1,0,1,0,1,0,1,0,1],
     [0,1,0,1,0,1,0,1,0,1],
     [0,1,0,1,0,1,0,1,0,1],
     [0,1,0,1,0,1,0,1,0,1],
     [0,0,0,1,0,0,0,1,0,0],
     [1,1,1,1,1,1,1,1,1,0]],

    # z=2
    [[1,1,1,1,1,1,1,1,1,1],
     [0,0,0,0,0,0,0,0,0,1],
     [1,1,1,1,1,1,1,1,0,1],
     [0,0,0,0,0,0,0,0,0,1],
     [0,1,1,1,1,1,1,1,1,1],
     [0,0,0,0,0,0,0,0,0,1],
     [1,1,1,1,1,1,1,1,0,1],
     [0,0,0,0,0,0,0,0,0,1],
     [0,1,1,1,1,1,1,1,1,0],
     [1,1,1,1,1,1,1,1,1,0]],

    # z=3
    [[1,1,1,1,1,1,1,1,1,1],
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,1,1,1,1,1,1,1,1],
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,1,1,1,1,1,1,1,1],
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,1,1,1,1,1,1,1,1],
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,1,1,1,1,1,1,1,0],
     [1,1,1,1,1,1,1,1,0,0]],

    # z=4
    [[1,1,1,1,1,1,1,1,1,1],
     [1,0,0,0,0,0,0,0,0,1],
     [1,0,1,1,1,1,1,1,0,1],
     [1,0,1,0,0,0,0,1,0,1],
     [1,0,1,0,1,1,0,1,0,1],
     [1,0,1,0,1,1,0,1,0,1],
     [1,0,1,0,0,0,0,1,0,0],
     [1,0,1,1,1,1,1,1,0,1],
     [1,0,0,0,0,0,0,0,0,0],
     [1,1,1,1,1,1,1,1,1,0]],

    # z=5
    [[1,1,0,1,1,0,1,1,1,1],
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,1,0,1,0,1,1,1,1],
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,1,0,1,0,1,1,1,1],
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,0,1,1,0,1,1,1,1],
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,1,1,1,1,1,1,1,0],
     [1,1,1,1,1,1,1,1,1,0]],

    # z=6
    [[1,1,1,1,1,1,1,1,1,1],
     [1,0,0,0,0,1,0,0,0,1],
     [1,0,1,1,0,1,0,1,0,1],
     [1,0,0,1,0,1,0,1,0,1],
     [1,0,0,1,0,1,0,1,0,1],
     [1,1,0,1,0,1,0,1,0,1],
     [1,1,0,1,0,1,0,1,0,0],
     [1,1,0,1,0,1,0,1,0,0],
     [1,1,0,0,0,1,0,0,0,1],
     [1,1,1,1,1,1,0,1,1,0]],

    # z=7
    [[1,1,1,1,1,1,1,1,1,1],
     [1,0,0,0,0,0,0,0,0,1],
     [1,0,1,1,1,1,1,1,0,1],
     [0,0,1,0,0,0,0,1,0,0],
     [1,0,1,0,1,1,0,1,0,1],
     [1,0,1,0,1,1,0,1,0,1],
     [1,0,1,0,0,0,0,1,0,0],
     [1,0,1,1,1,1,1,1,0,1],
     [1,0,0,0,0,0,0,0,0,1],
     [1,1,1,1,1,1,1,0,0,0]],

    # z=8
    [[1,1,1,1,1,1,1,1,1,1],
     [0,0,0,0,0,0,0,0,0,1],
     [1,1,1,1,1,1,1,1,0,1],
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,1,1,0,1,1,0,1,0],
     [0,0,0,0,0,0,0,0,0,1],
     [1,1,0,1,0,1,0,1,0,1],
     [0,0,0,0,0,0,0,0,0,0],
     [1,1,1,0,1,0,1,0,0,0],
     [1,0,0,1,1,1,1,1,0,0]],

    # z=9
    [[0,1,1,1,0,1,1,1,1,1],
     [0,0,1,1,0,0,1,1,1,0],
     [1,0,1,0,1,0,1,1,1,0],
     [1,0,0,0,1,0,0,0,0,0],
     [1,0,0,0,0,0,1,0,0,0],
     [1,1,1,0,0,1,1,0,1,0],
     [1,1,1,0,0,1,1,0,1,0],
     [1,1,1,1,1,0,0,0,1,0],
     [0,1,1,0,1,0,1,0,0,0],
     [1,1,1,0,0,0,0,1,0,0]]
])

#maze[x, y, z]
#maze[0] z layer
#maze[0][0] y layer
maze = np.transpose(maze_data, (2,1,0))
# print("شکل ماز بعد از transpose:", maze.shape)
# print("اولین ردیف لایه z=0:", maze[:, 0, 0].tolist())
start = (0,0,0)
end = (9,9,9)


# def load_maze_from_file(filename="Maze3D-Corrected.txt"):
#     maze = np.zeros((10,10,10), dtype=int)
#     z = -1
#     with open(filename , 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
            
#             if line.startswith('# Layer') or 'z' in line:
#                 z += 1
#                 y = 0
#                 continue
            
#             if line.startswith('[') and line.endswith(']'):
#                 #datas
#                 row = [int(x) for x in line[1:-1].replace('',' ').split(',')]
#                 if len(row) == 10 and y<10 and z< 10:
#                     maze[y, :,z] = row
#                 y +=1
                
#     return maze
# maze = load_maze_from_file('Maze3D-Corrected.txt')


#initializing setting:
population_size = 500
chromosome_length = 50
generations = 500
mutation_rate = 0.02
crossover_rate = 0.85
tournoment_size = 5


#each chromosom is a series of directions 
directions = [(0,0,1) , (0,0,-1) , (0,1,0) , (0,-1,0) , (1,0,0) , (-1,0,0)]
def fitness(individual):
    x , y , z = start
    visited = set([(x,y,z)])
    len_path = 0
    
    for gene in individual:
        dx , dy , dz = directions[gene]
        nx , ny , nz = x + dx , y + dy , z + dz
        
        if  0 <= nx < 10 and 0 <= ny < 10 and 0 <= nz < 10:
            
        
            if maze[nx , ny , nz] == 0 and (nx , ny , nz) not in visited:
                x , y , z = nx , ny , nz
                visited.add((x , y , z))
                len_path += 1
                
                if (x , y ,z) == end:
                    return 200000 + (1000 - len_path) * 10
            
        
        #fixed #manhatan distance
    distance_to_end = abs(x- 9) + abs(y - 9) + abs(z- 9)
    return 1000 - (distance_to_end * 10)
        
def create_invidual():
    return [random.randint(0,5) for _ in range(chromosome_length)]
population = [create_invidual() for _ in range(population_size)]

def tournament_selection(population , fitnesses):
    competitors = random.sample(range(len(population)) , tournoment_size)
    best = max(competitors , key = lambda i: fitnesses[i])
    return population[best]

def mutate(invidual):
    for i in range(len(invidual)):
        if random.random() < mutation_rate:
            invidual[i] = random.randint(0,5)
    return invidual

def crossover(parent1 , parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, 49)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1 , child2
    return parent1 , parent2


# genetic algorithm and solution

best = None
best_path = []
best_fitness = -1

population = [create_invidual() for _ in range(population_size)]

start_time = time.time()
print("start solving...")

for gene in range(generations):
    fitnesses =  [fitness(ind) for ind in population]
    
    
    curr_best_fitness = max(fitnesses)
    curr_best_index = fitnesses.index(curr_best_fitness)
    curr_best = population[curr_best_index]
    
    if gene % 50 == 0:
        print(f'generation {gene} with best fitness = {curr_best_fitness}')\
            
    if curr_best_fitness > 50000:
        print(f' solution found in generation { gene}')
        
        x , y ,z = start
        best_path = [(x,y,z)]
        for move in curr_best:
            dx , dy , dz = directions[move]
            nx , ny , nz = x + dx , y+ dy , z + dz
            if 0 <= nx < 10 and 0 <= ny < 10 and 0 <= nz < 10:
                if maze[nx, ny, nz] == 0 and (nx, ny, nz) not in best_path:
                    x, y, z = nx, ny, nz
                    best_path.append((x, y, z))
                    if (x, y, z) == end:
                        break
            
        break
    
    new_population = []
    # elitism
    new_population.append(population[curr_best_index])
    
    
    while len(new_population) < population_size:
        parent1 = tournament_selection(population , fitnesses)
        parent2 = tournament_selection( population,fitnesses)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.extend([child1, child2])
    
    population = new_population[:population_size]
    
print('\n finded path:')
for step in best_path:
    print(step)

end_time = time.time()
print(f"\ntime: {end_time - start_time:.2f}")