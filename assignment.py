# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:24:29 2025

@author: Dubari Kalita
"""

import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from numba import jit
import seaborn as sns

# ------------------------ Optimized Reward Function -------------------------

@jit(nopython=True)
def calculate_expected_reward(state, action, V, max_cars, rental_credit, move_cost,
                               gamma, modified, free_shuttle, parking_cost, parking_limit,
                               rental_probs_0, rental_probs_1, return_probs_0, return_probs_1, poisson_limit):
    cars_loc1, cars_loc2 = state

    # Move cars
    if action >= 0:
        cars_moved = min(action, cars_loc1)
        cars_loc1 -= cars_moved
        cars_loc2 = min(cars_loc2 + cars_moved, max_cars)
    else:
        cars_moved = min(-action, cars_loc2)
        cars_loc2 -= cars_moved
        cars_loc1 = min(cars_loc1 + cars_moved, max_cars)

    # Move cost
    if modified and action > 0:
        move_cost_total = max(0, cars_moved - free_shuttle) * move_cost
    else:
        move_cost_total = abs(cars_moved) * move_cost

    # Parking cost
    parking_cost_total = 0
    if modified:
        if cars_loc1 > parking_limit:
            parking_cost_total += parking_cost
        if cars_loc2 > parking_limit:
            parking_cost_total += parking_cost

    expected_reward = -move_cost_total - parking_cost_total

    for rental1 in range(poisson_limit + 1):
        for rental2 in range(poisson_limit + 1):
            for return1 in range(poisson_limit + 1):
                for return2 in range(poisson_limit + 1):
                    prob = rental_probs_0[rental1] * rental_probs_1[rental2] * return_probs_0[return1] * return_probs_1[return2]

                    rented1 = min(rental1, cars_loc1)
                    rented2 = min(rental2, cars_loc2)
                    reward = (rented1 + rented2) * rental_credit

                    new_cars1 = min(cars_loc1 - rented1 + return1, max_cars)
                    new_cars2 = min(cars_loc2 - rented2 + return2, max_cars)

                    expected_reward += prob * (reward + gamma * V[new_cars1, new_cars2])

    return expected_reward

# ------------------------- Jack Car Rental Class ---------------------------

class JackCarRental:
    def __init__(self, modified=False):
        self.max_cars = 20
        self.max_move = 5
        self.rental_credit = 10
        self.move_cost = 2
        self.gamma = 0.9
        self.theta = 1e-4

        self.rental_lambda = [3, 4]
        self.return_lambda = [3, 2]

        self.modified = modified
        self.free_shuttle = 1 if modified else 0
        self.parking_cost = 4 if modified else 0
        self.parking_limit = 10 if modified else float('inf')

        self.poisson_limit = 12
        self.rental_probs = [self._poisson_probs(lam) for lam in self.rental_lambda]
        self.return_probs = [self._poisson_probs(lam) for lam in self.return_lambda]

        self.V = np.zeros((self.max_cars + 1, self.max_cars + 1))
        self.policy = np.zeros((self.max_cars + 1, self.max_cars + 1), dtype=int)

    def _poisson_probs(self, lam):
        probs = [poisson.pmf(n, lam) for n in range(self.poisson_limit)]
        probs.append(1 - sum(probs))
        return np.array(probs)

    def _expected_reward(self, state, action):
        return calculate_expected_reward(
            state, action, self.V, self.max_cars, self.rental_credit,
            self.move_cost, self.gamma, self.modified, self.free_shuttle,
            self.parking_cost, self.parking_limit,
            self.rental_probs[0], self.rental_probs[1],
            self.return_probs[0], self.return_probs[1],
            self.poisson_limit
        )

    def policy_evaluation(self):
        delta = float('inf')
        iteration = 0

        while delta > self.theta:
            delta = 0
            new_V = np.zeros_like(self.V)

            for i in range(self.max_cars + 1):
                for j in range(self.max_cars + 1):
                    old_v = self.V[i, j]
                    action = self.policy[i, j]
                    new_v = self._expected_reward((i, j), action)
                    new_V[i, j] = new_v
                    delta = max(delta, abs(old_v - new_v))

            self.V = new_V
            iteration += 1

        return iteration

    def policy_improvement(self):
        policy_stable = True

        for i in range(self.max_cars + 1):
            for j in range(self.max_cars + 1):
                old_action = self.policy[i, j]

                best_action = 0
                best_value = float('-inf')

                max_move_from_1 = min(self.max_move, i)
                max_move_from_2 = min(self.max_move, j)

                for action in range(-max_move_from_2, max_move_from_1 + 1):
                    value = self._expected_reward((i, j), action)
                    if value > best_value:
                        best_value = value
                        best_action = action

                self.policy[i, j] = best_action

                if old_action != best_action:
                    policy_stable = False

        return policy_stable

    def solve(self):
        iteration = 0
        policy_stable = False

        print("Starting policy iteration...")

        while not policy_stable:
            eval_iterations = self.policy_evaluation()
            print(f"Policy Iteration {iteration + 1}: Evaluation converged in {eval_iterations} iterations")
            policy_stable = self.policy_improvement()
            iteration += 1

            if iteration > 20:
                print("Maximum iterations reached")
                break

        print(f"Policy iteration converged in {iteration} iterations")
        return self.policy, self.V

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(self.policy, annot=True, cmap='RdBu_r', center=0,
                    ax=ax1, cbar_kws={'label': 'Cars moved'})
        ax1.set_title('Optimal Policy')
        ax1.set_xlabel('Cars at location 2')
        ax1.set_ylabel('Cars at location 1')

        sns.heatmap(self.V, cmap='viridis', ax=ax2, cbar_kws={'label': 'Value'})
        ax2.set_title('Value Function')
        ax2.set_xlabel('Cars at location 2')
        ax2.set_ylabel('Cars at location 1')

        plt.tight_layout()
        plt.show()

# ----------------------------- Run the Program -----------------------------

if __name__ == "__main__":
    import time

    print("Solving original Jack's Car Rental problem...")
    start = time.time()
    jack_original = JackCarRental(modified=False)
    policy_orig, value_orig = jack_original.solve()
    print("Time taken (original):", time.time() - start)
    jack_original.plot_results()

    print("\nSolving modified Jack's Car Rental problem...")
    start = time.time()
    jack_modified = JackCarRental(modified=True)
    policy_mod, value_mod = jack_modified.solve()
    print("Time taken (modified):", time.time() - start)
    jack_modified.plot_results()
