import NSGA_optimization as Multiprocess_NSGA  # 在这里更改调换包
import time
import config
import json
import random

# 是否使用并行遗传算法
Multiprocessing = True

T, PAYLOAD = config.get_para("T&p")
SHOVEL_TRUCK, SHOVEL_DUMP = config.get_para("match")
DUMPS, SHOVELS, TRUCK_NUM = config.get_para("object")
SHOVEL_TARGET_WORKLOAD, DUMP_TARGET_WORKLOAD = config.get_para("target")
LOADING_TIME, UNLOADING_TIME = config.get_para("time")
PATH_LENGTH = config.get_para("path")

BOARD_INTERVAL = 10  # 板车决策粒度

SHOVEL_CURPRO = {}
for i in SHOVELS:
    SHOVEL_CURPRO[i] = 0

# 算法参数
POPULATION_SIZE = 35  # 正常是35*7,目前敏感性改成10*4
ITERATION_NUM = 400  # 原始1000,400

# 定义要实验的算法
# ALGORITHMS = ["NSLNS_evolution", "NSGA_evolution", "ALNS_evolution", "VNS_evolution"]
ALGORITHMS = ["NSLNS_evolution"] #["NSGA_evolution", "ALNS_evolution", "VNS_evolution"]
if __name__ == "__main__":
    if Multiprocessing:
        for algorithm_name in ALGORITHMS:
            for seed in range(0, 1):  # 使用随机种子0-10
                print(seed)
                random.seed(seed)  # 设置随机种子
                # 初始化算法实例
                algorithm = Multiprocess_NSGA.creation(
                    T, PAYLOAD, TRUCK_NUM, SHOVELS, DUMPS,
                    SHOVEL_TARGET_WORKLOAD, SHOVEL_TRUCK, SHOVEL_DUMP,
                    LOADING_TIME, UNLOADING_TIME, PATH_LENGTH, BOARD_INTERVAL,
                    POPULATION_SIZE, ITERATION_NUM
                )

                start = time.time()
                algorithm.initialization()

                # 根据算法名调用对应的进化方法
                if hasattr(algorithm, algorithm_name):
                    result = getattr(algorithm, algorithm_name)()  # 动态调用算法方法
                else:
                    raise AttributeError(f"Algorithm '{algorithm_name}' not found in Multiprocess_NSGA.")

                # 生成结果文件名
                file_name = f"result2/{algorithm_name}_{seed}.json"
                with open(file_name, 'w') as f:
                    json.dump(result, f, indent=4)

                print(f"{algorithm_name} with seed {seed} completed in {time.time() - start:.2f} seconds.")
