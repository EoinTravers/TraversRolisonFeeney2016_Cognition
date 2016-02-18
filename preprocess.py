#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from squeak import *
import os
import matplotlib.pyplot as plt
from numpy.random import normal, choice
pd.options.mode.chained_assignment = None # Disable pandas warning

# Import data
data = pd.read_csv('data/raw_data.csv', sep=";")
data = data.sort('id')

# Code questions
from stimuli import *
# Even subject numbers did list A, odds numbers list B
odd = [nr % 2 for nr in data.subject_nr]
data['list'] = [['B', 'A'][nr] for nr in odd] # A if odd, B if even

code_lists = {'A':code_listA, 'B':code_listB}
question_lists = {'A':question_listA, 'B':question_listB}
response_lists = {'A':response_listA, 'B':response_listB}

data['code'] = [code_lists[lst][number] for lst, number in zip(data.list, data.stimuli_number)]

data['responses'] = [response_lists[lst][number] for lst, number in zip(data.list, data.stimuli_number)]
data['correct'] = data.responses.map(lambda r: r[0])
data['heuristic'] = data.responses.map(lambda r: r[1])
data['other1'] = data.responses.map(lambda r: r[2])
data['other2'] = data.responses.map(lambda r: r[3])

data['condition'] = data.code.map(lambda c: c[0])

def get_choice(response, correct, heuristic, other1, other2):
	if response == correct:
		return 'correct'
	elif response == heuristic:
		return 'heuristic'
	elif response == other1:
		return 'other1'
	elif response == other2:
		return 'other2'
	else:
		raise ValueError

data['choice'] = [get_choice(response, correct, heuristic, other1, other2) for
					response, correct, heuristic, other1, other2 in zip(data.response, data.correct, data.heuristic, data.other1, data.other2)]
data['acc'] = data.choice=='correct'

data['is_heuristic'] = data.choice == 'heuristic'
subject_bias = data[data.condition=='C'].groupby('subject_nr')['is_heuristic'].mean()
data['bias'] = [subject_bias.loc[s] for s in data.subject_nr]

# Data screening
# Exclude participants who didn't finish the experiment
print "%i participants initially." % len(set(data.subject_nr))

trials_per_subject = data.groupby('subject_nr')['choice'].aggregate(len)
missing_trials = trials_per_subject != 8
data['missing_trials'] = [missing_trials[nr] for nr in data.subject_nr]

n0 = len(data)
data = data[~ data.missing_trials]
n1 = len(data)
print 'Dropping %i trials from %i participants who \
didn\'t complete the experiment.' % ((n0 - n1), missing_trials.sum())


# Exclude trials over 100 seconds
n0 = len(data)
data = data[data.rt < 100000]
n1 = len(data)
dropped = float(n0 - n1)
print 'Dropping %i trials over 100 seconds (%.3f%%).' % ((n0 - n1), 100*float(n0 - n1)/n0)

# Reindex
data.index = range(len(data))

# Parse mouse data
data['x'] = data.xList.map(list_from_string) - (data.w*.5)
data['y'] = (data.yList.map(list_from_string) - (data.w*.5))
data['t'] = data.tList.map(list_from_string)

# Rotate so top left -> Top
#from cmath import phase
#p = phase(complex(-1, -1))
#xy = zip(*[rotate(x, y, p) for x, y in zip(data.x, data.y)])
#data['x'] = xy[0]
#data['y'] = xy[1]
#del xy

# Scale to 0 -> 1
data['x'] = [x/abs(x[-1]) for x in data.x]
data['y'] = [y/abs(y[-1]) for y in data.y]

# Deal with miscoding of no-conflict problem 4 (item code B4)
# The correct response was coded as the heuristic, and vice versa, so 0 and 1 are the
# wrong way around in that permutation codes for these items.
# To solve this, I change the permutation number (0-15) for each item
# to be the one that gives the right coding,
# for instance changeing 0 ([0,1,2,3]) for 4 ([1,0,2,3])


permutation_mappings = [4, 5, 6, 7, 0, 1, 2, 3, 9, 8, 11, 10, 13, 12, 15, 14] # <- Swap these values
## 					    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15    <- ...for these ones
for i in range(len(data)):
	if data.code.iloc[i] == 'B4':
		data.perm.iloc[i] = permutation_mappings[data.perm.iloc[i]]

## Code for making sure the above has worked
# data['end_x' ] = data.x.map(lambda x: x[-1])
# data['end_y' ] = data.y.map(lambda x: x[-1])
# d = data[(data.choice=='correct') | (data.choice=='heuristic')]
# from scipy.stats import mode
# def isProblem(trial):
# 	cohort = data[(data.condition==trial.condition) &
# 					(data.choice==trial.choice) &
# 					(data.perm==trial.perm)]
# 	probx = trial.end_x != mode(cohort.end_x)[0][0]
# 	proby = trial.end_y != mode(cohort.end_y)[0][0]
# 	return probx | proby
# d['bad'] = [isProblem(trial) for i, trial in d.iterrows()]
# b = d[d.bad]
# b.head()


## The complicated bit ##

## location of the response options on each trial was determined by
## randomly selecting one of the permutations below. These were
## constrained so that the correct option was always adjacent to the
## heuristic.  In order to analyse the time course data, the data
## recorded on each trial must be renormalized to a standard co-ordinate
## space --- arbitrarily, I've mapped the correct response option to the
## top left (that is, [-1, 1]), the heuristic to the top right ([1,1];
## remember that python places positive y values on the top) and the two
## other responses to the bottom corners ([-1, -1], [-1,1]).  Because of
## the way the permutations have been done, it is not possible to
## consistently remap the other filler response options consistently -
## what's coded as 'other1' on some trials is 'other2' on
## others. However, this has no bearing on any of the analysis, and
## could only be avoided by not permutating the responses, which would
## have been a much bigger problem.

## The permutateions were as follows:
##permutations = [[ 0, 1, 2, 3],# (0)
##				 [  0, 1, 3, 2],# (1)
##				 [  0, 2, 3, 1],# (2)
##				 [  0, 3, 2, 1],# (3)
##				 [  1, 0, 2, 3],# (4)
##				 [  1, 0, 3, 2],# (5)
##				 [  1, 2, 3, 0],# (6)
##				 [  1, 3, 2, 0],# (7)
##				 [  2, 0, 1, 3],# (8)
##				 [  2, 1, 0, 3],# (9)
##				 [  2, 3, 0, 1],# (10)
##				 [  2, 3, 1, 0],# (11)
##				 [  3, 0, 1, 2],# (12)
##				 [  3, 1, 0, 2],# (13)
##				 [  3, 2, 0, 1],# (14)
##				 [  3, 2, 1, 0]]# (15)
## This is a clockwise order, from top left around
## 0 = correct response, 1 = heuristic, 2 and 3 = other responses

# Monster code to remap data from each possible permutation
print "Translating trajectory data onto common co-ordinate space."

p = np.pi*.5 # pi/2, useful when rotating co-ordinates
for i in range(len(data)):
	perm = data.perm.iloc[i]
	if perm == 0 or perm == 1:
		pass
	elif perm == 2 or perm == 3:
		# Rotate anticlockwise once, then flip x
		x, y = rotate(data['x'].iloc[i], data['y'].iloc[i], p)
		x = x*-1
		data['x'].iloc[i] = x
		data['y'].iloc[i] = y
	elif perm == 4 or perm == 5:
		# Flip on x axis
		data['x'].iloc[i] = data['x'].iloc[i] * -1
	elif perm == 6 or perm == 7:
		# Rotate anticlockwise once
		x, y = rotate(data['x'].iloc[i], data['y'].iloc[i], p)
		data['x'].iloc[i] = x
		data['y'].iloc[i] = y
	elif perm == 8 or perm == 12:
		# Anticlockwise 3 times
		x, y = rotate(data['x'].iloc[i], data['y'].iloc[i], p*3)
		data['x'].iloc[i] = x
		data['y'].iloc[i] = y
	elif perm == 9 or perm == 13:
		# Rotate anticlockwise once, then flip y
		x, y = rotate(data['x'].iloc[i], data['y'].iloc[i], p)
		y = y*-1
		data['x'].iloc[i] = x
		data['y'].iloc[i] = y
	elif perm == 10 or perm == 14:
		# Rotate anticlockwise twice
		x, y = rotate(data['x'].iloc[i], data['y'].iloc[i], p*2)
		data['x'].iloc[i] = x
		data['y'].iloc[i] = y
	elif perm == 11 or perm == 15:
		# Flip on y axis
		data['y'].iloc[i] = data['y'].iloc[i] * -1

## Flip everything on the y axis, so it's:
## Top left = correct
## Top right = heuristic
## Bottom = Others
data['y'] = data.y.map(lambda y: y*-1)


## Use this function to ensure that the trajectories have been
## remapped correctly. All correct trajectories (green) should end in
## the top left, heuristic ones (red) in the rop right, and others
## (blue) in the bottom corners.
def plot_ends(n=-1):
	end_x = data.x.map(lambda x: x[-1]) + normal(0, .1, size=len(data))
	end_y = data.y.map(lambda x: x[-1]) + normal(0, .1, size=len(data))
	choice = data.choice
	color_map = {'correct':'g', 'heuristic':'r', 'other1':'b', 'other2':'b'}
	color = np.array([color_map[c] for c in choice])
	plt.figure(figsize=(10,10))
	if n==-1:
        ## Plot every permutation
		Ps = range(16)
	elif type(n) == 'int':
        ## Plot the one given
		Ps = [n]
	else:
        ## Plot the list given
		Ps = n
	for p in Ps:
		plt.subplot(4,4,p+1)
		I = data[data.perm==p].index
		for i in I:
			plt.plot(end_x[i], end_y[i], 'o',  color=color[i])
		plt.title(p)
	plt.show()
# plot_ends()

print "Calculating summary statistics from trajectory data..."
## Summary statistics on mouse data

## Time normalised trajectories are not useful here, but if required...
# data['nx'], data['ny'] = zip(*[even_time_steps(x, y, t) for x, y, t in zip(data.x, data.y, data.t)])
# nx = pd.concat(list(data.nx), axis=1).T
# ny = pd.concat(list(data.ny), axis=1).T

## Instead, using standard time, interpolated to 20 msec intervals
## (the default of `uniform_time`) to deal with any variation in the
## sample rate
data['rx'] = [uniform_time(x, t, max_duration=100000) for x, t in zip(data.x, data.t)]
data['ry'] = [uniform_time(y, t, max_duration=100000) for y, t in zip(data.y, data.t)]
rx = pd.concat(list(data.rx), axis=1).T
ry = pd.concat(list(data.ry), axis=1).T

## Get velocity in 20ms windows
rvel = [np.sqrt(rx.iloc[i].diff()**2 + ry.iloc[i].diff()**2) for i in range(len(rx))]
rvel = pd.concat(rvel, axis=1).T
rvel = rvel.drop([0.0], axis=1) # NaN velocity for first step.
rvel.to_csv('data/rvel.csv', index=False)

## Plot to get a sense of how many seperate 'movements' in each trial
## This plots the velocity profile for 32 random trajectories.

# I = np.random.choice(rvel.index, 32)
# plot_to = 30000 # Only show first 30 seconds
# R = rvel.iloc[I]
# R = R[R.columns[R.columns < plot_to]]
# plt.figure(figsize=(14, 14))
# a = 1
# for i, r in R.iterrows():
# 	plt.subplot(8,4, a)
# 	plt.plot(R.columns/1000, r, 'b')
# 	a += 1
# plt.show()

## Path length (divide by sqrt(2) so that minimum possible = 1)
data['path_length'] = rvel.apply(np.sum, 1) / np.sqrt(2)

## We'll say the cursor is moving if it traverses more than .01 (that
## is .005 of the standardized window width) within a 20 msec
## interval.  (Threshold based on plots of data)
is_moving = rvel > .01

# Next, take these sequences of moving / not-moving time slices,
# and count how many actual movements there are, where movement is
# a sequence of 5 or more slices (100 msec) in motion,
# broken by 100 msec or more not in motion.
def count_movements(seq, window_size=5):
	# make sure all runs are well-bounded
	bounded = np.hstack(([0], seq, [0]))
	difs = np.diff(bounded)
	run_starts, = np.where(difs > 0)
	run_ends, = np.where(difs < 0)
	moving_length = run_ends - run_starts
	not_moving_length = run_starts[1:] - run_ends[:-1]
	a = run_starts[0]
	not_moving_length = np.concatenate([[a], not_moving_length])
	full_movements = (moving_length > window_size) & (not_moving_length > window_size)
	return (np.sum(full_movements))
data['movements'] = is_moving.apply(count_movements, 1)

## How close does the cursor pass to the competing response option?
def min_distance(x, y, refX, refY):
	distance = np.sqrt( (refX - x) ** 2 + (refY - y)**2)
	return np.min(distance)

min_correct = lambda i: min_distance(rx.iloc[i], ry.iloc[i], -1, 1)
min_heuristic = lambda i: min_distance(rx.iloc[i], ry.iloc[i], 1, 1)
# (Divide by 2 to scale in terms of screen width)
min_dist_from_correct = [min_correct(i)/2 for i in data.index]
min_dist_from_heuristic = [min_heuristic(i)/2 for i in data.index]

data['proximity_to_correct'] = min_dist_from_correct
data['proximity_to_heuristic'] = min_dist_from_heuristic

def get_prox_to_other(trial):
	if trial['choice'] == 'heuristic':
		return trial['proximity_to_correct']
	elif trial['choice'] == 'correct':
		return trial['proximity_to_heuristic']
	else:
		return None
data['proximity_to_other'] = data.apply(lambda trial: get_prox_to_other(trial), axis=1)

## Cleaning up
## Save only these variables
vars_to_keep = ['id', 'response', 'rt', 'perm', 'code', 'condition',
				'stimuli_number', 'subject_nr', 'trial', 'bias',
				'correct', 'heuristic', 'choice', 'acc',
				'path_length', 'movements', 'proximity_to_other']
clean_data = data[vars_to_keep]

## Save
clean_data.to_csv('data/processed.csv', index=False)


## # # # # Convert to long data  # #

print "Converting time course data to 'long' format."

# Store the measures of interest in a 'wide' format
wide_r = data[[u'id', 'condition', u'response', u'rt',  u'subject_nr', 'bias', u'trial',
					u'stimuli_number', u'choice', u'acc', 'perm']]
steps = rx.columns
steps = steps[steps < 60000] # Only look at first 60 seconds
# Do a sparse sample now, only looking at every Nth time step
# Look at every 5th sample, so a 100 msec interval (increase this number to save your processor)
use_every = 5
steps = steps[::use_every]
for i in steps:
	wide_r['rx_%i' % i] = rx[i]
	wide_r['ry_%i' % i] = ry[i]

# Melt into a 'long' format
id_vars = ['subject_nr', 'trial', 'condition', 'choice', 'acc',
		   'stimuli_number', 'bias', 'perm']
long_r = pd.melt(wide_r, id_vars=id_vars, value_vars = ['rx_%i' % i for i in steps])
long_r['step'] = long_r.variable.map(lambda s: int(s[3:]))
long_r['rx'] = long_r.value

## Repeat melting process for additional dependant variables
## (in this case, just 'ry', but loop allows the same to be done with any time course measure).
## I think there may be a way of achieving this withing the 'melt' above.
for var in ['ry']: # Add more variables as needed
	tmp_long = pd.melt(wide_r, id_vars = id_vars, value_vars = ['%s_%i' % (var, i) for i in steps])
	long_r[var] = tmp_long.value
long_r = long_r.drop(['variable', 'value'], axis=1)

## Code according to region of the screen
## There are four regions - TL, TR, BL, and BR.

def toss():
	## If on border between quadrants, toss a coin.
	return choice([-.001,.001], 1)[0]

def get_section(x, y):
	if x==0:
		x = x + toss()
	if y==0:
		y = y + toss()
	if x < 0 and y > 0:
		return 'correct' # Top left
	elif x > 0 and y > 0:
		return 'heuristic' # Top right
	elif x < 0 and y < 0:
		return 'other1' # bottom left (although it doesn't matter)
	elif x > 0 and y < 0:
		return 'other2' # bottom right

long_r['section'] = [get_section(x, y) for x, y in zip(long_r.rx, long_r.ry)]
for r in ['correct', 'heuristic', 'other1', 'other2']:
	long_r[r] = long_r.section == r

## Distance from each response option
## (suggested by reviewer)
long_r['distance_from_correct'] = np.sqrt(np.power(-1 - long_r.rx, 2) +
									np.power(1 - long_r.ry, 2))
long_r['distance_from_heuristic']	= np.sqrt(np.power(1 - long_r.rx, 2) +
									np.power(1 - long_r.ry, 2))
long_r['distance_from_other1']  = np.sqrt(np.power(-1 - long_r.rx, 2) +
									np.power(-1 - long_r.ry, 2))
long_r['distance_from_other2']  = np.sqrt(np.power(1 - long_r.rx, 2) +
									np.power(-1 - long_r.ry, 2))

long_r.to_csv(os.path.join('data', 'long.csv'), index=False)

## Save each condition seperately (optional)
conflict = long_r[long_r.condition == 'C']
baseline = long_r[long_r.condition == 'B']
conflict.to_csv(os.path.join('data', 'long_conflict.csv'), index=False)
baseline.to_csv(os.path.join('data', 'long_baseline.csv'), index=False)
print 'Done!'
