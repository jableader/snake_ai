import pygame
from pygame.locals import *
from random import randint, seed
import math
import os, sys
import neat
import visualize

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

def mult_coords(s, c):
	return (c[0] * s, c[1] * s)

def add_coords(*args):
	sy, sx = 0, 0
	for y, x in args:
		sy += y
		sx += x
	return (sy, sx)


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

	return 1 / (2**(count - 1))

class Board:
	def __init__(self, boardsize = GAME_WIDTH):
		self.heading = DIR_NORTH
		self.done = False
		self.boardsize = boardsize

		self._snake = [(self.boardsize // 2, self.boardsize // 2)]
		self._apples = set()
		self._score = 0

		self.add_apple()
		self._last_apple_distance = self.get_apple_distance()

	def set_heading(self, dir):
		if dir != opposite_direction(self.heading): # Can't run back on yourself
			self.heading = dir

	def step(self):
		if self.done:
			return

		next_head = add_coords(self._snake[-1], heading_coords(self.heading))
		if self.is_oob(next_head) or next_head in self._snake:
			self.done = True
			return

		delta = self.get_apple_distance() - self._last_apple_distance
		max_boardsize = 2 * self.boardsize * 1.42
		bonus = delta / max_boardsize
		if bonus < 0:
			bonus *=2
		self._score += bonus

		self._snake.append(next_head)
		if next_head in self._apples:
			self._apples.remove(next_head)
			self.add_apple()
		else:
			self._snake.pop(0)

	def add_apple(self):
		loc = self._snake[0] # Just force it into the loop
		while loc in self._snake or loc in self._apples:
			loc = (randint(0, self.boardsize - 1), randint(0, self.boardsize - 1))
		self._apples.add(loc)

	def render(self, screen):
		screen.fill((0, 0, 0))

		dx, dy = SCREEN_WIDTH // self.boardsize, SCREEN_HEIGHT // self.boardsize
		for y, x in self._snake:
			surf = pygame.Surface((dx, dy))
			surf.fill((255, 255, 255))
			screen.blit(surf, (x * dx, y * dy))

		self.render_eyes(screen, dx, dy)

		for y, x in self._apples:
			apple_center = (int(dx * (x + 0.5)), int(dy * (y + 0.5)))
			pygame.draw.circle(screen, (255, 0, 0), apple_center, dx//2)

		self.render_params(screen)
		pygame.display.flip()

	def render_eyes(self, screen, dx, dy):
		head_center = add_coords(self._snake[-1], (0.5, 0.5))
		offset_in_dir = lambda start, dir, amount: add_coords(start, mult_coords(amount, heading_coords(dir)))

		eye_center = offset_in_dir(head_center, self.heading, 0.35)
		left_eye = offset_in_dir(eye_center, left_dir(self.heading), 0.25)
		right_eye = offset_in_dir(eye_center, right_dir(self.heading), 0.25)

		draw_eye = lambda ey, ex: pygame.draw.circle(screen, (255, 0, 0), (int(ex * dx), int(ey * dy)), int(0.1*dx))
		draw_eye(*left_eye)
		draw_eye(*right_eye)

	def render_params(self, screen):
		sz = 10
		font = pygame.font.SysFont('Comic Sans MS', sz)
		color = (0, 200, 0)
		renderables = list(self.get_named_params()) + [('Score', self.get_score())]
		for i, nameValue in enumerate(renderables):
			text = font.render("%s: %f" % nameValue, True, color)
			screen.blit(text, (5, 5 + 1.3*sz * i))

	@staticmethod
	def param_names():
		return [
			"head_fwd_is_death",
			"head_left_is_death",
			"head_right_is_death",
			"distance",
			"theta left",
			"theta right",
			]

	def get_named_params(self):
		return tuple(zip(Board.param_names(), self.get_params()))

	def get_params(self):
		apple = next(iter(self._apples))
		head = self._snake[-1]
		fwd = self.heading
		left = left_dir(self.heading)
		right = right_dir(self.heading)
		is_death = lambda loc: loc != head and (loc in self._snake or self.is_oob(loc))
		is_apple = lambda loc: loc in self._apples

		return [
			dist_scale(head, fwd, is_death),
			dist_scale(head, left, is_death),
			dist_scale(head, right, is_death),
			dist_euclids(head, apple),
			angle_ratio_points(head, apple, left),
			angle_ratio_points(head, apple, right)
			]

	def is_oob(self, pt):
		y, x = pt
		return y < 0 or y >= self.boardsize or x < 0 or x >= self.boardsize

	def get_apple_distance(self):
		hy, hx = self._snake[-1]
		ay, ax = next(iter(self._apples))

		return ((hx - ax)**2 + (hy - ay)**2)**0.5

	def get_score(self):
		return 10*len(self._snake) + self._score

def show_game(board, run_events, delay=5):
	pygame.init()
	pygame.font.init() # you have to call this at the start,

	screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
	board.render(screen)
	while not board.done:
		run_events(board)
		board.step()
		pygame.time.delay(delay)
		board.render(screen)

def wait_for_space():
	while True:
		ev = pygame.event.poll()
		if ev.type == KEYDOWN:
			if ev.key == K_SPACE:
				return
			elif ev.key == K_ESCAPE:
				sys.exit(0)
		elif ev.type == QUIT:
			sys.exit(0)

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
				elif event.key == K_SPACE:
					wait_for_space()
				return # Only process one keypress per step

	show_game(Board(), run_events, 200)

def get_next_heading(net, board):
	output = net.activate(board.get_params())

	best = max(output)
	if output[0] == best:
		return left_dir(board.heading)
	if output[1] == best:
		return board.heading
	else:
		return right_dir(board.heading)

def eval_net(net, seeds):
	score = 0
	for s in seeds:
		b = Board(GAME_WIDTH)
		snake_len, count = 1, 0
		while not b.done and count < b.boardsize * snake_len / 2:
			if len(b._snake) > snake_len:
				count, snake_len = 0, len(b._snake)
			else:
				count += 1

			b.set_heading(get_next_heading(net, b))
			b.step()
		score += b.get_score()
	return score

suspected_generation = -1
def eval_genomes(genomes, config):
	global suspected_generation
	suspected_generation += 1
	seeds = [randint(0, 999999) for x in range(50)]

	for genome_id, genome in genomes:
		net = neat.nn.RecurrentNetwork.create(genome, config)
		genome.fitness = eval_net(net, seeds)

def update_from_net(b, net):
		b.set_heading(get_next_heading(net, b))
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit(0)
			elif event.type == KEYDOWN:
				if event.key == K_SPACE:
					wait_for_space()
				elif event.key == K_RETURN:
					b.done = True

def train(generations):
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
	winner = p.run(eval_genomes, generations)

	# Display the winning genome.
	print('\nBest genome:\n{!s}'.format(winner))

	visualize_net(config, winner, stats)

	net = neat.nn.RecurrentNetwork.create(winner, config)
	while True:
		show_game(Board(), lambda b: update_from_net(b, net), 50)
		wait_for_space()

def visualize_net(config, winner, stats):
	node_names = {}
	for i, name in enumerate(Board.param_names()):
		node_names[(i + 1) * - 1] = name

	visualize.draw_net(config, winner, True, node_names=node_names)
	visualize.plot_stats(stats, ylog=False, view=True)
	visualize.plot_species(stats, view=True)

task = 'train'
generations = 10

if len(sys.argv) > 1:
	task = sys.argv[1]

if len(sys.argv) >= 2:
	generations = sys.argv[2]

if task == 'train':
	train(generations)
elif task == 'play':
	play()
elif task == 'checkpoint':
	from_checkpoint(generations)
