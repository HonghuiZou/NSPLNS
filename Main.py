"""
Author: Honghui Zou, Kaiqi Zhao
E_mail: hhzou@buaa.edu.cn
"""

import NSPLNS_optimization as Multiprocess_NSPLNS
import time
import config
import json
import random

T, PAYLOAD = config.get_para("T&p")
SHOVEL_TRUCK, SHOVEL_DUMP = config.get_para("match")
DUMPS, SHOVELS, TRUCK_NUM = config.get_para("object")
LOADING_TIME, UNLOADING_TIME = config.get_para("time")
PATH_LENGTH = config.get_para("path")

Multiprocessing = True
BOARD_INTERVAL = 10
SHOVEL_CURPRO = {}
for i in SHOVELS:
    SHOVEL_CURPRO[i] = 0
POPULATION_SIZE = 35
ITERATION_NUM = 1000


ALGORITHMS = ["NSPLNS_evolution"]
if __name__ == "__main__":
    if Multiprocessing:
        for algorithm_name in ALGORITHMS:
            for seed in range(0,10):
                # print(algorithm_name)
                # print(seed)
                random.seed(seed)
                algorithm = Multiprocess_NSPLNS.creation(
                    T, PAYLOAD, TRUCK_NUM, SHOVELS, DUMPS, SHOVEL_TRUCK, SHOVEL_DUMP,
                    LOADING_TIME, UNLOADING_TIME, PATH_LENGTH, BOARD_INTERVAL,
                    POPULATION_SIZE, ITERATION_NUM,algorithm_name,seed
                )

                start = time.time()
                algorithm.initialization()

                if hasattr(algorithm, algorithm_name):
                    result = getattr(algorithm, algorithm_name)()
                else:
                    raise AttributeError(f"Algorithm '{algorithm_name}' not found in Multiprocess_NSPLNS.")

                file_name = f"{algorithm_name}_{seed}.json"
                with open(file_name, 'w') as f:
                    json.dump(result, f, indent=4)

                print(f"{algorithm_name} with seed {seed} completed in {time.time() - start:.2f} seconds.")
