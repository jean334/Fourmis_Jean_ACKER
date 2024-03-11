"""
Module managing an ant colony in a labyrinth.
"""
import numpy as np
import maze_p
from maze_p import Maze
import pheromone_p
from pheromone_p import Pheromon
import direction as d
import pygame as pg
from mpi4py import MPI
import matplotlib.pyplot as plt
import re

UNLOADED, LOADED = False, True

exploration_coefs = 0.


class Colony:
    """
    Represent an ant colony. Ants are not individualized for performance reasons!

    Inputs :
        nb_ants  : Number of ants in the anthill
        pos_init : Initial positions of ants (anthill position)
        max_life : Maximum life that ants can reach
    """
    def __init__(self, nb_ants, pos_init, max_life):
        # Each ant has is own unique random seed
        self.seeds = np.arange(1, nb_ants+1, dtype=np.int64)
        # State of each ant : loaded or unloaded
        self.is_loaded = np.zeros(nb_ants, dtype=np.int8)
        # Compute the maximal life amount for each ant :
        # Updating the random seed :
        self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
        # Amount of life for each ant = 75% à 100% of maximal ants life
        self.max_life = max_life * np.ones(nb_ants, dtype=np.int32)
        self.max_life -= np.int32(max_life*(self.seeds/2147483647.))//4
        # Ages of ants : zero at beginning
        self.age = np.zeros(nb_ants, dtype=np.int16)
        # History of the path taken by each ant. The position at the ant's age represents its current position.
        self.historic_path = np.zeros((nb_ants, max_life+1, 2), dtype=np.int16)
        self.historic_path[:, 0, 0] = pos_init[0]
        self.historic_path[:, 0, 1] = pos_init[1]
        # Direction in which the ant is currently facing (depends on the direction it came from).
        self.directions = d.DIR_NONE*np.ones(nb_ants, dtype=np.int16)
        """
        self.sprites = []
        img = pg.image.load("ants.png").convert_alpha()
        for i in range(0, 32, 8):
            self.sprites.append(pg.Surface.subsurface(img, i, 0, 8, 8))
        """

    def return_to_nest(self, loaded_ants, pos_nest, food_counter):
        """
        Function that returns the ants carrying food to their nests.

        Inputs :
            loaded_ants: Indices of ants carrying food
            pos_nest: Position of the nest where ants should go
            food_counter: Current quantity of food in the nest

        Returns the new quantity of food
        """
        self.age[loaded_ants] -= 1

        in_nest_tmp = self.historic_path[loaded_ants, self.age[loaded_ants], :] == pos_nest #les fourmis qui sont au
        #nid sont les fourmis dont la position précédente est le nid (la position à l'âge précédent)
        if in_nest_tmp.any():
            in_nest_loc = np.nonzero(np.logical_and(in_nest_tmp[:, 0], in_nest_tmp[:, 1]))[0]
            if in_nest_loc.shape[0] > 0:
                in_nest = loaded_ants[in_nest_loc]
                self.is_loaded[in_nest] = UNLOADED
                self.age[in_nest] = 0
                food_counter += in_nest_loc.shape[0]
        return food_counter

    def explore(self, unloaded_ants, the_maze, pos_food, pos_nest, pheromones):
        """
        Management of unloaded ants exploring the maze.

        Inputs:
            unloadedAnts: Indices of ants that are not loaded
            maze        : The maze in which ants move
            posFood     : Position of food in the maze
            posNest     : Position of the ants' nest in the maze
            pheromones  : The pheromone map (which also has ghost cells for
                          easier edge management)

        Outputs: None
        """
        # Update of the random seed (for manual pseudo-random) applied to all unloaded ants
        self.seeds[unloaded_ants] = np.mod(16807*self.seeds[unloaded_ants], 2147483647)

        # Calculating possible exits for each ant in the maze:
        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze_p.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze_p.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze_p.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze_p.WEST) > 0

        # Reading neighboring pheromones:
        north_pos = np.copy(old_pos_ants)
        north_pos[:, 1] += 1
        north_pheromone = pheromones.pheromon[north_pos[:, 0], north_pos[:, 1]]*has_north_exit

        east_pos = np.copy(old_pos_ants)
        east_pos[:, 0] += 1
        east_pos[:, 1] += 2
        east_pheromone = pheromones.pheromon[east_pos[:, 0], east_pos[:, 1]]*has_east_exit

        south_pos = np.copy(old_pos_ants)
        south_pos[:, 0] += 2
        south_pos[:, 1] += 1
        south_pheromone = pheromones.pheromon[south_pos[:, 0], south_pos[:, 1]]*has_south_exit

        west_pos = np.copy(old_pos_ants)
        west_pos[:, 0] += 1
        west_pheromone = pheromones.pheromon[west_pos[:, 0], west_pos[:, 1]]*has_west_exit

        max_pheromones = np.maximum(north_pheromone, east_pheromone)
        max_pheromones = np.maximum(max_pheromones, south_pheromone)
        max_pheromones = np.maximum(max_pheromones, west_pheromone)

        # Calculating choices for all ants not carrying food (for others, we calculate but it doesn't matter)
        choices = self.seeds[:] / 2147483647.

        # Ants explore the maze by choice or if no pheromone can guide them:
        ind_exploring_ants = np.nonzero(
            np.logical_or(choices[unloaded_ants] <= exploration_coefs, max_pheromones[unloaded_ants] == 0.))[0]
        if ind_exploring_ants.shape[0] > 0:
            ind_exploring_ants = unloaded_ants[ind_exploring_ants]
            valid_moves = np.zeros(choices.shape[0], np.int8)
            nb_exits = has_north_exit * np.ones(has_north_exit.shape) + has_east_exit * np.ones(has_east_exit.shape) + \
                has_south_exit * np.ones(has_south_exit.shape) + has_west_exit * np.ones(has_west_exit.shape)
            while np.any(valid_moves[ind_exploring_ants] == 0):
                # Calculating indices of ants whose last move was not valid:
                ind_ants_to_move = ind_exploring_ants[valid_moves[ind_exploring_ants] == 0]
                self.seeds[:] = np.mod(16807*self.seeds[:], 2147483647)
                # Choosing a random direction:
                dir = np.mod(self.seeds[ind_ants_to_move], 4)
                old_pos = self.historic_path[ind_ants_to_move, self.age[ind_ants_to_move], :]
                new_pos = np.copy(old_pos)
                new_pos[:, 1] -= np.logical_and(dir == d.DIR_WEST,
                                                has_west_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 1] += np.logical_and(dir == d.DIR_EAST,
                                                has_east_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 0] -= np.logical_and(dir == d.DIR_NORTH,
                                                has_north_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                new_pos[:, 0] += np.logical_and(dir == d.DIR_SOUTH,
                                                has_south_exit[ind_ants_to_move]) * np.ones(new_pos.shape[0], dtype=np.int16)
                # Valid move if we didn't stay in place due to a wall
                valid_moves[ind_ants_to_move] = np.logical_or(new_pos[:, 0] != old_pos[:, 0], new_pos[:, 1] != old_pos[:, 1])
                # and if we're not in the opposite direction of the previous move (and if there are other exits)
                valid_moves[ind_ants_to_move] = np.logical_and(
                    valid_moves[ind_ants_to_move],
                    np.logical_or(dir != 3-self.directions[ind_ants_to_move], nb_exits[ind_ants_to_move] == 1))
                # Calculating indices of ants whose move we just validated:
                ind_valid_moves = ind_ants_to_move[np.nonzero(valid_moves[ind_ants_to_move])[0]]
                # For these ants, we update their positions and directions
                self.historic_path[ind_valid_moves, self.age[ind_valid_moves] + 1, :] = new_pos[valid_moves[ind_ants_to_move] == 1, :]
                self.directions[ind_valid_moves] = dir[valid_moves[ind_ants_to_move] == 1]

        ind_following_ants = np.nonzero(np.logical_and(choices[unloaded_ants] > exploration_coefs,
                                                       max_pheromones[unloaded_ants] > 0.))[0]
        if ind_following_ants.shape[0] > 0:
            ind_following_ants = unloaded_ants[ind_following_ants]
            self.historic_path[ind_following_ants, self.age[ind_following_ants] + 1, :] = \
                self.historic_path[ind_following_ants, self.age[ind_following_ants], :]
            max_east = (east_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 1] += \
                max_east * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_west = (west_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 1] -= \
                max_west * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_north = (north_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 0] -= max_north * np.ones(ind_following_ants.shape[0], dtype=np.int16)
            max_south = (south_pheromone[ind_following_ants] == max_pheromones[ind_following_ants])
            self.historic_path[ind_following_ants, self.age[ind_following_ants]+1, 0] += max_south * np.ones(ind_following_ants.shape[0], dtype=np.int16)

        # Aging one unit for the age of ants not carrying food
        if unloaded_ants.shape[0] > 0:
            self.age[unloaded_ants] += 1

        # Killing ants at the end of their life:
        ind_dying_ants = np.nonzero(self.age == self.max_life)[0]
        if ind_dying_ants.shape[0] > 0:
            self.age[ind_dying_ants] = 0
            self.historic_path[ind_dying_ants, 0, 0] = pos_nest[0]
            self.historic_path[ind_dying_ants, 0, 1] = pos_nest[1]
            self.directions[ind_dying_ants] = d.DIR_NONE

        # For ants reaching food, we update their states:
        ants_at_food_loc = np.nonzero(np.logical_and(self.historic_path[unloaded_ants, self.age[unloaded_ants], 0] == pos_food[0],
                                                     self.historic_path[unloaded_ants, self.age[unloaded_ants], 1] == pos_food[1]))[0]
        if ants_at_food_loc.shape[0] > 0:
            ants_at_food = unloaded_ants[ants_at_food_loc]
            self.is_loaded[ants_at_food] = True

    def advance(self, the_maze, pos_food, pos_nest, pheromones, food_counter=0):
        old_pheromones = pheromones.pheromon.copy()
        loaded_ants = np.nonzero(self.is_loaded == True)[0]
        unloaded_ants = np.nonzero(self.is_loaded == False)[0]
        if loaded_ants.shape[0] > 0:
            food_counter = self.return_to_nest(loaded_ants, pos_nest, food_counter)
        if unloaded_ants.shape[0] > 0:
            self.explore(unloaded_ants, the_maze, pos_food, pos_nest, pheromones)

        old_pos_ants = self.historic_path[range(0, self.seeds.shape[0]), self.age[:], :]
        has_north_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze_p.NORTH) > 0
        has_east_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze_p.EAST) > 0
        has_south_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze_p.SOUTH) > 0
        has_west_exit = np.bitwise_and(the_maze.maze[old_pos_ants[:, 0], old_pos_ants[:, 1]], maze_p.WEST) > 0
        # Marking pheromones:
        
        [pheromones.mark(old_pheromones, self.historic_path[i, self.age[i], :],
                         [has_north_exit[i], has_east_exit[i], has_west_exit[i], has_south_exit[i]]) for i in range(self.directions.shape[0])]
        return food_counter

    @classmethod
    def display(self, screen, sprites, directions, historic_path, age):

        [screen.blit(sprites[directions[i]], (8*historic_path[i, age[i], 1], 8*historic_path[i, age[i], 0])) for i in range(directions.shape[0])]


def inserer_valeur_dans_fichier_markdown(valeur,fichier_markdown, regex, balise_debut, balise_fin):
    with open(fichier_markdown, 'r') as f:
        contenu = f.read()
        occurrences = re.findall(r'{}{}{}'.format(balise_debut, regex, balise_fin), contenu)
        print(occurrences)
    for occurrence in occurrences:
        print(occurrence)
        contenu = contenu.replace("{}".format(occurrence), 
                                  "{}{}{}".format(balise_debut, valeur, balise_fin))
    
    with open(fichier_markdown, 'w') as f:
        f.write(contenu)


if __name__ == "__main__":
    import sys
    import time

    comm = MPI.COMM_WORLD
    comGlobal = MPI.COMM_WORLD.Dup()
    rank      = comGlobal.rank
    nbp       = comGlobal.size
    nb_iter = int(sys.argv[1])

    newComm = comm.Split(rank!=0)
    newRank = newComm.rank

    size_laby = 25, 25
    if len(sys.argv) > 2:
        size_laby = int(sys.argv[1]),int(sys.argv[2])
    
    nb_ants = size_laby[0]*size_laby[1]//4
    total_ants = size_laby[0]*size_laby[1]//4
    max_life = 500
    nb_ants_per_processus = None
    nb_historic_path = None
    fps_buff = None
    global_buffer = None
    temps_calcul = None
    temps_envoie = None
    temps_total  = None
    temps_calcul_unit = None
    temps_total_unit = None

    if rank == 0:
        pg.init()
        resolution = size_laby[1]*8, size_laby[0]*8
        screen = pg.display.set_mode(resolution)
        img = pg.image.load("cases.png").convert_alpha()
        cases_img = []
        
        for i in range(0, 128, 8):
            cases_img.append(pg.Surface.subsurface(img, i, 0, 8, 8))
        
        sprites = []
        img = pg.image.load("ants.png").convert_alpha()
        for i in range(0, 32, 8):
            sprites.append(pg.Surface.subsurface(img, i, 0, 8, 8))
        
        snapshop_taken = False
        maze_buff = np.empty(size_laby, dtype=np.uint8)
        comm.Recv(maze_buff, source=1, tag=0)
        nb_ants_per_processus = np.empty(nbp-1, dtype=np.int16)
        comm.Recv(nb_ants_per_processus, source=1, tag=1)

        pherom_buff = np.empty((size_laby[0]+2, size_laby[1]+2), dtype=np.double)

        global_buffer = np.empty(total_ants+total_ants*(max_life+1)*2+total_ants+1, dtype=np.int16)
        temps_affichage = np.empty(nb_iter, dtype=np.float64)
        nb_ants = 0

    
    if rank >= 1:
        if len(sys.argv) > 3:
            max_life = int(sys.argv[3])
        
        pos_food = size_laby[0]-1, size_laby[1]-1
        pos_nest = 0, 0
        a_maze = maze_p.Maze(size_laby, 12345)
        
        nb_ants = nb_ants//(nbp-1)
        if total_ants % (nbp-1) != 0 and rank <= total_ants % (nbp-1):
            nb_ants +=1
            ants = Colony(nb_ants, pos_nest, max_life)
        else:
            ants = Colony(nb_ants, pos_nest, max_life)
        
        if rank == 1:
            comm.Send(a_maze.maze, dest=0, tag=0)
            nb_ants_per_processus = np.empty(nbp-1, dtype=np.int16)         
            temps_reception = np.empty(nb_iter, dtype=np.float32)
            comm.Send(nb_ants_per_processus, dest=0, tag=1)
            food_counter_buff = np.empty(nb_iter, dtype=np.int32)

            temps_calcul = np.empty(nb_iter, dtype=np.float64)
            temps_envoie = np.empty(nb_iter, dtype=np.float64)
            temps_total  = np.empty(nb_iter, dtype=np.float64)
            temps_affichage  = np.empty(nb_iter, dtype=np.float64)
            
            temps_calcul_unit = np.empty(1, dtype=np.float64)
            temps_total_unit = np.empty(1, dtype=np.float64)

        food_counter_buff_unit = np.zeros(1, dtype=np.int32)
        newComm.Gather(sendbuf=np.array([nb_ants], dtype=np.int16), recvbuf=nb_ants_per_processus, root=0) 
        
        if rank == 1:
            nb_historic_path = np.array([nb_ants_per_processus[i]*2*(max_life+1) for i in range(nbp-1)])
        global_buffer = np.empty(total_ants+total_ants*(max_life+1)*2+total_ants+1, dtype=np.int16)

        unloaded_ants = np.array(range(nb_ants))
        alpha = 0.9
        beta  = 0.99
        if len(sys.argv) > 4:
            alpha = float(sys.argv[4])
        if len(sys.argv) > 5:
            beta = float(sys.argv[5])
        pherom = pheromone_p.Pheromon(size_laby, pos_food, alpha, beta)
        

        pherom_buff = np.empty((size_laby[0]+2, size_laby[1]+2), dtype=np.double)
        food_counter = 0


    for i in range(nb_iter):
        if rank == 0:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit(0)

            comm.Recv(pherom_buff, source=1, tag=2)
            comm.Recv(global_buffer, source=1, tag=3)
            food_counter_buff = global_buffer[-1]
            t1 = time.time()
            Pheromon.display(screen, pherom_buff)
            mazeImg = Maze.display(maze_buff, cases_img)
            screen.blit(mazeImg, (0, 0))

            Colony.display(screen, sprites, global_buffer[:total_ants], 
                           global_buffer[total_ants:total_ants+total_ants*(max_life+1)*2].reshape((total_ants,max_life+1,2)), 
                           global_buffer[total_ants+total_ants*(max_life+1)*2:total_ants+total_ants*(max_life+1)*2+total_ants])
            pg.display.update()
            t2 = time.time()


            if food_counter_buff == 1 and not snapshop_taken:
                pg.image.save(screen, "MyFirstFood.png")
                snapshop_taken = True
            temps_affichage[i] = t2-t1


         

        if rank >= 1:
            deb = time.time()
            newComm.Allreduce(pherom.pheromon,pherom_buff,  MPI.MAX) 
            newComm.Gatherv(sendbuf= ants.directions, 
                    recvbuf=(global_buffer, nb_ants_per_processus), root=0)
            newComm.Gatherv(sendbuf=ants.historic_path.flatten(), 
                    recvbuf=(global_buffer[total_ants:], nb_historic_path), root=0)
            newComm.Gatherv(sendbuf=ants.age, 
                    recvbuf=(global_buffer[total_ants+total_ants*(max_life+1)*2:], nb_ants_per_processus), root=0)
            t3 = time.time()
            food_counter = ants.advance(a_maze, pos_food, pos_nest, pherom, food_counter)
            pherom.do_evaporation(pos_food)
            end = time.time()

            newComm.Reduce(np.array(food_counter, dtype=np.int16), global_buffer[total_ants+total_ants*(max_life+1)*2+total_ants:], root=0, op=MPI.SUM)
            newComm.Reduce(np.array(end-deb, dtype=np.float64), temps_total_unit, root=0, op=MPI.SUM)
            newComm.Reduce(np.array(end-t3, dtype=np.float64), temps_calcul_unit, root=0, op=MPI.SUM)
        
        if rank == 1:
            t4 = time.time()
            comm.Send(pherom_buff, dest=0, tag=2)
            comm.Send(global_buffer, dest=0, tag=3)
            t5 = time.time()
            temps_envoie[i] = t5-t4
            temps_calcul[i] = temps_calcul_unit[0]
            temps_total[i] = temps_total_unit[0]+ t5-t4
            food_counter_buff[i] = food_counter_buff_unit[-1]
            print(f"FPS : {1./((temps_total[i])):6.2f}, nourriture : {food_counter_buff[i]:7d}", end='\r')
    

    """
    if rank==0:     
        comm.Send(temps_affichage, dest=1, tag=200)

    if rank==1:
        comm.Recv(temps_affichage, source=0,tag=200)
        temps_calcul /= (nbp-1)
        temps_total /= (nbp-1)
        X = np.arange(1,nb_iter+1)
        selection = np.arange(1,nb_iter+1, 100)
        plt.title("Les différents temps de calcul par itération (nbp = {})".format(nbp))
        plt.ylabel("temps de calcul")
        plt.xlabel("itérations")
        plt.plot(X[selection[10:]], temps_calcul[selection[10:]], label="temps de calcul par moyen")
        plt.plot(X[selection[10:]], temps_envoie[selection[10:]], label="temps d'envoi des données")
        plt.plot(X[selection[10:]], temps_total[selection[10:]], label="temps total par itération")
        plt.plot(X[selection[10:]], temps_affichage[selection[10:]], label="temps d'affichage par itération")
        plt.legend()
        inserer_valeur_dans_fichier_markdown("{}".format(np.mean(1./temps_total[selection])), "Rapport_Fourmis_Jean_ACKER.md", 
                                         r'\d*.*\d*', "<fps2{}>".format(nbp), "</fps2{}>".format(nbp))  
        inserer_valeur_dans_fichier_markdown("{}".format(np.mean(temps_calcul)), "Rapport_Fourmis_Jean_ACKER.md", 
                                         r'\d*', "<tempscalcul2{}>".format(nbp), "</tempscalcul2{}>".format(nbp))    
        inserer_valeur_dans_fichier_markdown("{}".format(np.mean(temps_envoie)), "Rapport_Fourmis_Jean_ACKER.md", 
                                         r'\d*', "<tempsenvoie2{}>".format(nbp), "</tempsenvoie2{}>".format(nbp))    
        inserer_valeur_dans_fichier_markdown("{}".format(np.mean(temps_total)), "Rapport_Fourmis_Jean_ACKER.md", 
                                         r'\d*', "<tempstotal2{}>".format(nbp), "</tempstotal2{}>".format(nbp))    
        inserer_valeur_dans_fichier_markdown("{}".format(food_counter_buff[-1]), "Rapport_Fourmis_Jean_ACKER.md", 
                                         r'\d*', "<nourriture2{}>".format(nbp), "</nourriture2{}>".format(nbp))
        inserer_valeur_dans_fichier_markdown("{}".format(np.mean(temps_affichage)), "Rapport_Fourmis_Jean_ACKER.md", 
                                         r'\d*', "<affichage2{}>".format(nbp), "</affichage2{}>".format(nbp))

        plt.savefig("./ressources/fps_plot_p2{}.png".format(nbp))  
        """
