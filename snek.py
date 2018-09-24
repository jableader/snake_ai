import pygame
from pygame.locals import *
from random import randint

import os
import neat

GAME_WIDTH = 40

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

DIR_NORTH = 0
DIR_EAST = 1
DIR_SOUTH = 2
DIR_WEST = 3

def opposite_direction(dir):
    return (dir + 2) % 4

def left_dir(dir):
    return (dir - 1) % 4

def right_dir(dir):
    return (dir + 1) % 4

def heading_coords(dir):
    if dir == DIR_NORTH:
        return (-1, 0)
    elif dir == DIR_EAST:
        return (0, 1)
    elif dir == DIR_SOUTH:
        return (1, 0)
    elif dir == DIR_WEST:
        return (0, -1)

def add_coords(c1, c2):
    return (c1[0] + c2[0], c1[1] + c2[1])

def is_oob(pt):
    y, x = pt
    return y < 0 or y >= GAME_WIDTH or x < 0 or x >= GAME_WIDTH

def dist_scale(location, heading, is_end_fn):
    count = 0
    dir_coords = heading_coords(heading)
    while not is_end_fn(location):
        location = add_coords(location, dir_coords)
        count += 1

    return (count - GAME_WIDTH) / GAME_WIDTH

class Board:
    def __init__(self):
        self.heading = DIR_NORTH
        self.done = False

        self._snake = [(GAME_WIDTH // 2, GAME_WIDTH // 2)]
        self._apples = set()

        self.add_apple()

    def set_heading(self, dir):
        if dir != opposite_direction(self.heading): # Can't run back on yourself
            self.heading = dir

    def step(self):
        if self.done:
            return

        next_head = add_coords(self._snake[-1], heading_coords(self.heading))
        if is_oob(next_head) or next_head in self._snake:
            self.done = True
            return

        self._snake.append(next_head)
        if next_head in self._apples:
            self._apples.remove(next_head)
            self.add_apple()
        else:
            self._snake.pop(0)

    def add_apple(self):
        loc = self._snake[0] # Just force it into the loop
        while loc in self._snake or loc in self._apples:
            loc = (randint(0, GAME_WIDTH - 1), randint(0, GAME_WIDTH - 1))
        self._apples.add(loc)

    def render(self, screen):
        dx, dy = SCREEN_WIDTH // GAME_WIDTH, SCREEN_HEIGHT // GAME_WIDTH
        for y, x in self._snake:
            surf = pygame.Surface((dx, dy))
            surf.fill((255, 255, 255))
            screen.blit(surf, (x * dx, y * dy))

        for y, x in self._apples:
            surf = pygame.Surface((dx, dy))
            surf.fill((255, 0, 0))
            screen.blit(surf, (x * dx, y * dy))

    def get_params(self):
        head = self._snake[-1]
        fwd = self.heading
        left = left_dir(self.heading)
        right = right_dir(self.heading)
        is_snake = lambda loc: loc in self._snake
        is_apple = lambda loc: loc in self._apples
        return (
            dist_scale(head, fwd, is_oob),
            dist_scale(head, left, is_oob),
            dist_scale(head, right, is_oob),
            dist_scale(head, fwd, is_snake),
            dist_scale(head, left, is_snake),
            dist_scale(head, right, is_snake),
            dist_scale(head, fwd, is_apple),
            dist_scale(head, left, is_apple),
            dist_scale(head, right, is_apple))

    def get_score(self):
        hy, hx = self._snake[-1]
        ay, ax = next(iter(self._apples[0]))

        apple_dist = ((hx - ax)**2 + (hy - ay)**2)**0.5
        max_dist = GAME_WIDTH * 1.4142
        return len(self._snake) + (apple_dist / max_dist)

def play():
    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    board = Board()
    while not board.done:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    board.set_heading(DIR_NORTH)
                elif event.key == K_RIGHT:
                    board.set_heading(DIR_EAST)
                elif event.key == K_DOWN:
                    board.set_heading(DIR_SOUTH)
                elif event.key == K_LEFT:
                    board.set_heading(DIR_WEST)
                elif event.key == K_ESCAPE:
                    board.done = True

        board.step()
        screen.fill((0, 0, 0))
        board.render(screen)
        pygame.display.flip()
        pygame.time.delay(100)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        b = Board()
        snake_len, cnt = 1, 0
        cnt = 0
        while not b.done:
            output = net.activate(b.get_params())
            best = max(output)
            if output[0] == best:
                b.set_heading(left_dir(b.heading))
            elif output[2] == best:
                b.set_heading(right_dir(b.heading))
            b.step()

            newlen = len(b._snake)
            count += 1
            if newlen > snake_len:
                count, snake_len = 0,
            elif count > 50:
                break 

        genome.fitness = b.get_score()

def train():
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config.cfg')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 10)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

train()
