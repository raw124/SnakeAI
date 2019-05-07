from snakeGame import Game as game
import pygame
from pygame.locals import *
import random

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import numpy as np

LR = 1e-3
goal_steps = 300
score_requirement = 50
initial_games = 5000


def some_random_games_first():
    # Each of these is its own game.
    for episode in range(10):

        env = game()
        env.reset()
        start = True
        for _ in range(goal_steps):
            action = random.randrange(0, 3)

            if start:
                start = False
                action = 2

            # do it! render the previous view
            env.render()
            observation, reward, done, info = env.step(action)
            if done: break


def generate_population(model):
    global score_requirement

    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        env = game()
        env.reset()

        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            if len(prev_observation) == 0:
                action = random.randrange(0, 3)
            else:
                if not model:
                    action = random.randrange(0, 3)
                else:
                    prediction = model.predict(prev_observation.reshape(-1, len(prev_observation), 1))
                    action = np.argmax(prediction[0])

            # do it!
            observation, reward, done, info = env.step(action)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done: break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)

                action_sample = [0, 0, 0]
                action_sample[data[1]] = 1
                output = action_sample
                # saving our training data
                training_data.append([data[0], output])

        # save overall scores
        scores.append(score)

    if len(accepted_scores) != 0:
        score_requirement = mean(accepted_scores)

    # just in case you wanted to reference later
    training_data_save = np.array([training_data, score_requirement])
    np.save('saved.npy', training_data_save)

    return training_data


def create_dummy_model(training_data):
    shape_second_parameter = len(training_data[0][0])
    x = np.array([i[0] for i in training_data])
    X = x.reshape(-1, shape_second_parameter, 1)
    y = [i[1] for i in training_data]
    model = create_neural_network_model(input_size=len(X[0]), output_size=len(y[0]))
    return model


def create_neural_network_model(input_size, output_size):
    network = input_data(shape=[None, input_size, 1], name='input')
    network = tflearn.fully_connected(network, 32)
    network = tflearn.fully_connected(network, 32)
    network = fully_connected(network, output_size, activation='softmax')
    network = regression(network, name='targets')
    model = tflearn.DNN(network, tensorboard_dir='tflearn_logs')

    return model


def train_model(training_data, model=False):
    shape_second_parameter = len(training_data[0][0])
    x = np.array([i[0] for i in training_data])
    X = x.reshape(-1, shape_second_parameter, 1)
    y = [i[1] for i in training_data]

    model.fit({'input': X}, {'targets': y}, n_epoch=10, batch_size=16, show_metric=True)
    model.save('minisnake_trained.tflearn')

    return model


def evaluate(model):
    # now it's time to evaluate the trained model
    scores = []
    choices = []
    foodCounts = []
    for each_game in range(5):
        score = 0
        game_memory = []
        prev_obs = []
        env = game()
        env.reset()
        count = 0
        foodCount = 0
        currentScore = 0
        stop = False
        while stop == False:
            if foodCount < 5:
                if count == 150:
                    count = 0
                    currentScore = 0
                    if currentScore <= 150:
                        stop = True
            elif foodCount < 10:
                if count == 300:
                    count = 0
                    currentScore = 0
                    if currentScore <= 300:
                        stop = True
            elif foodCount < 15:
                if count == 600:
                    count = 0
                    currentScore = 0
                    if currentScore <= 600:
                        stop = True
            else:
                if count == 900:
                    count = 0
                    currentScore = 0
                    if currentScore <= 900:
                        stop = True
                
            env.render()

            if len(prev_obs) == 0:
                action = random.randrange(0, 3)
            else:
                prediction = model.predict(prev_obs.reshape(-1, len(prev_obs), 1))
                action = np.argmax(prediction[0])

            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            currentScore += reward
            if reward == 100:
                foodCount += 1
            count += 1
            if done: break

        scores.append(score)
        foodCounts.append(foodCount)

    print('\nAverage Score:', sum(scores) / len(scores))
    print('Average Food Eaten:', sum(foodCounts) / len(foodCounts))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
    print('Score Requirement:', score_requirement)

if __name__ == "__main__":
    generation = 1
    num = 0
    some_random_games_first()
    training_data = generate_population(None)
    model = create_dummy_model(training_data)
    while num != "3":
        print("\n1. Train Model.")
        print("2. Evaluate Model.")
        print("3. Quit.")
        num = input("Enter number: ")
        if num == "1":
            if generation == 1:
                model = train_model(training_data, model)
            else:
                print("\nSimulating games...")
                data = generate_population(None)
                if len(data) == 0:
                    print("\nNone of the simulated games reached the score requirement.\n")
                else:
                    training_data = np.append(training_data, data, axis=0)
                    model = train_model(training_data, model)
            print('Generation: ', generation, ' Initial Population: ', len(training_data), ' Score Requirement: ', score_requirement)
            generation += 1
        if num == "2":
            evaluate(model)
        if num == "3":
            print("Quit")