import pygame
from pygame.locals import *
from random import randint, seed
import math
import os, sys
import neat

GAME_WIDTH = 20

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

def angle_ratio_points(a, b, dir):
	y1, x1 = a
	y2, x2 = b

	angle = math.atan2(y2 - y1, x2 - x1) + math.pi
	if dir == DIR_EAST:
		angle += math.pi
	elif dir == DIR_NORTH:
		angle -= math.pi / 2
	elif dir == DIR_SOUTH:
		angle += math.pi / 2
	angle = angle % (2 * math.pi)
	angle -= math.pi
	ratio = abs(angle / math.pi)
	return ratio

def dist_euclids(a, b):
	ay, ax = a
	by, bx = b
	distance = ((ax - bx)**2 + (ay - by)**2)**0.5
	max_dist = GAME_WIDTH * (2**0.5)
	return (max_dist - distance) / max_dist

def dist_scale(location, heading, is_end_fn):
	count = 0
	dir_coords = heading_coords(heading)
	while (not is_end_fn(location)) and count <= GAME_WIDTH:
		location = add_coords(location, dir_coords)
		count += 1

	return (GAME_WIDTH - count) / GAME_WIDTH

class Board:
	def __init__(self):
		self.heading = DIR_NORTH
		self.done = False

		self._snake = [(GAME_WIDTH // 2, GAME_WIDTH // 2)]
		self._apples = set()

		self.add_apple()
		# seed(6969)

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

		self.render_params(screen)

	def render_params(self, screen):
		sz = 10
		font = pygame.font.SysFont('Comic Sans MS', sz)
		color = (0, 200, 0)
		for i, nameValue in enumerate(self.get_named_params()):
			text = font.render("%s: %f" % nameValue, True, color)
			screen.blit(text, (5, 5 + 1.3*sz * i))

	def get_named_params(self):
		head = self._snake[-1]
		fwd = self.heading
		left = left_dir(self.heading)
		right = right_dir(self.heading)
		is_snake = lambda loc: loc != head and loc in self._snake
		is_apple = lambda loc: loc in self._apples

		apple = next(iter(self._apples))
		return (
			("head_fwd_is_oob", dist_scale(head, fwd, is_oob)),
			("head_left_is_oob", dist_scale(head, left, is_oob)),
			("head_right_is_oob", dist_scale(head, right, is_oob)),
			("head_fwd_is_snake", dist_scale(head, fwd, is_snake)),
			("head_left_is_snake", dist_scale(head, left, is_snake)),
			("head_right_is_snake", dist_scale(head, right, is_snake)),
			("distance", dist_euclids(head, apple)),
			("theta left", angle_ratio_points(head, apple, left)),
			("theta right", angle_ratio_points(head, apple, right))
			)

	def get_params(self):
		return [x[1] for x in self.get_named_params()]

	def get_score(self):
		hy, hx = self._snake[-1]
		ay, ax = next(iter(self._apples))

		apple_dist = ((hx - ax)**2 + (hy - ay)**2)**0.5
		max_dist = GAME_WIDTH * 1.4142
		return 10*len(self._snake) + 1-(apple_dist / max_dist)

def show_game(board, run_events, delay=5):
	pygame.init()
	pygame.font.init() # you have to call this at the start, 

	screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
	while not board.done:
		screen.fill((0, 0, 0))
		board.render(screen)
		pygame.display.flip()

		run_events(board)
		board.step()
		pygame.time.delay(delay)

def play():
	def run_events(board):
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
				elif event.key==K_SPACE:
					input()

	show_game(Board(), run_events, 200)

def get_next_heading(net, board):
	output = net.activate(board.get_params())[0]

	if output < (1/3):
		return left_dir(board.heading)
	if output < (2/3):
		return board.heading
	else:
		return right_dir(board.heading)

def eval_net(net):
	b = Board()

	snake_len, count = 1, 0
	while not b.done and count < max(50, len(b._snake)*10):
		if len(b._snake) > snake_len:
			count, snake_len = 0, len(b._snake)
		else:
			count += 1

		b.set_heading(get_next_heading(net, b))
		b.step()
	return b.get_score()

def eval_genomes(genomes, config):
	for genome_id, genome in genomes:
		net = neat.nn.RecurrentNetwork.create(genome, config)
		genome.fitness = eval_net(net)

def update_blah(b, net):
		b.set_heading(get_next_heading(net, b))
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit(0)

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
	p.add_reporter(neat.Checkpointer(1000))

	# Run for up to N generations.
	winner = p.run(eval_genomes, 8000)
	print("winrar")

	net = neat.nn.RecurrentNetwork.create(winner, config)
	while input() != "q":
		show_game(Board(), lambda b: update_blah(b, net), 50)

	# Display the winning genome.
	print('\nBest genome:\n{!s}'.format(winner))

def from_checkpoint():
	# Load configuration.
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
						 neat.DefaultSpeciesSet, neat.DefaultStagnation,
						 'config.cfg')

	p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-29065')
	# Run for up to N generations.
	winner = p.run(eval_genomes, 10)
	print("winrar")

	net = neat.nn.RecurrentNetwork.create(winner, config)
	while input() != "q":
		show_game(Board(), lambda b: update_blah(b, net), 50)

train()
# play()
#from_checkpoint()