import random
import time
import copy
import numpy as np
import math
import Simulation_optimization as EMineSim
import config
import gc
import multiprocessing

MAXIMIZED = True
SWAPPING_TIME = 12
processes_number=4


class creation:
    def __init__(self, T, PAYLOAD, TRUCK_NUM, SHOVELS, DUMPS, SHOVEL_TRUCK,
                 SHOVEL_DUMP, LOADING_TIME, UNLOADING_TIME, PATH_LENGTH, BOARD_INTERVAL, POPULATION_SIZE,
                 ITERATION_NUM,algorithm_name,seed):

        self.T = T
        self.PAYLOAD = PAYLOAD
        self.TRUCK_NUM = TRUCK_NUM
        self.SHOVELS = SHOVELS
        self.DUMPS = DUMPS
        self.SHOVEL_TRUCK = SHOVEL_TRUCK
        self.SHOVEL_DUMP = SHOVEL_DUMP
        self.LOADING_TIME = LOADING_TIME
        self.UNLOADING_TIME = UNLOADING_TIME
        self.PATH_LENGTH = PATH_LENGTH

        self.BOARD_INTERVAL = BOARD_INTERVAL

        self.CHROMOSOME1_POPULATION = []
        self.CHROMOSOME2_POPULATION = []
        self.CHROMOSOME3_POPULATION = []

        self.population_size = POPULATION_SIZE
        self.generation = ITERATION_NUM
        self.algorithm_name=algorithm_name
        self.seed=seed

        length = {}
        max_length = 0
        for key, item in self.SHOVEL_TRUCK.items():
            target_dump = self.SHOVEL_DUMP[key]
            trip_time = self.LOADING_TIME[key] + self.PATH_LENGTH[(key, target_dump)][0] + self.UNLOADING_TIME[
                target_dump] + self.PATH_LENGTH[(target_dump, key)][0]
            length[key] = int(np.ceil(self.T / trip_time) + 1)
            if length[key] > max_length:
                max_length = length[key]

        self.chromosome3_length = 30
        self.chromosome1_length = max_length + self.chromosome3_length
        self.chromosome2_length = int(np.ceil(self.T / config.get_para('board_interval'))) * 2

    def initialization(self):
        chromosome1 = []
        chromosome3 = []
        for key, item in self.SHOVEL_TRUCK.items():
            for j in range(len(item)):
                chromosome_part = [1] * self.chromosome1_length
                chromosome1 += chromosome_part
                chromosome_part = [self.DUMPS.index(self.SHOVEL_DUMP[key]) for i in range(self.chromosome3_length)]
                chromosome3 += chromosome_part
        chromosome2 = []
        swapping_stations = self.DUMPS.copy()

        for i in range(len(swapping_stations)):
            chromosome_part = [1] * self.chromosome2_length
            chromosome2 += chromosome_part

        truck_repaired_schedule, car_repaired_schedule, swapper_arrivaltime, battery_car_time, fitness_value1, fitness_value2, fitness_value1_1, fitness_value1_2, fitness_value1_3,fitness_value2_1,fitness_value2_2,total_energy= EMineSim.Simulation(
            chromosome1, chromosome2, chromosome3, self.chromosome1_length, self.chromosome2_length, True)
        print(f"Initial_solution,obj1_values:{fitness_value1},obj2_values:{fitness_value2}")

        for i in range(self.population_size):
            random.shuffle(truck_repaired_schedule)
            self.CHROMOSOME1_POPULATION.append(truck_repaired_schedule.copy())
            random.shuffle(car_repaired_schedule)
            self.CHROMOSOME2_POPULATION.append(car_repaired_schedule.copy())
            self.CHROMOSOME3_POPULATION.append(chromosome3.copy())

    def task_crossover(self, x1, x2):
        x11 = copy.deepcopy(x1)
        x21 = copy.deepcopy(x2)
        schedule_len = len(x11)
        inherit_idx = random.sample(range(schedule_len), schedule_len // 2)
        for i in inherit_idx:
            x11[i], x21[i] = x21[i], x11[i]
        return x11, x21

    def operator_1(self, chromosome1, chromosome2, chromosome3, swapper_arrivaltime=0, battery_car_time=0):
        new_chromosome1 = copy.deepcopy(chromosome1)
        new_chromosome2 = copy.deepcopy(chromosome2)
        new_chromosome3 = copy.deepcopy(chromosome3)

        index = sorted(random.sample(range(0, len(chromosome1) - 1), int(len(chromosome1) * 0.1)))
        for i in index:
            new_chromosome1[i], new_chromosome1[i + 1] = new_chromosome1[i + 1], new_chromosome1[i]

        index = sorted(random.sample(range(0, len(chromosome2) - 1), int(len(chromosome2) * 0.1)))
        for i in index:
            new_chromosome2[i], new_chromosome2[i + 1] = new_chromosome2[i + 1], new_chromosome2[i]

        index = sorted(random.sample(range(0, len(chromosome3) - 1), int(len(chromosome3) * 0.1)))
        for i in index:
            new_chromosome3[i], new_chromosome3[i + 1] = new_chromosome3[i + 1], new_chromosome3[i]

        return new_chromosome1, new_chromosome2, new_chromosome3

    def operator_2(self, chromosome1, chromosome2, chromosome3, swapper_arrivaltime=0, battery_car_time=0):
        new_chromosome1 = copy.deepcopy(chromosome1)
        new_chromosome2 = copy.deepcopy(chromosome2)
        new_chromosome3 = copy.deepcopy(chromosome3)

        index = random.sample(range(0, len(chromosome1)), int(len(chromosome1) * 0.1))
        for idx in index:
            if new_chromosome1[idx] == 0:
                new_chromosome1[idx] = 1
            elif new_chromosome1[idx] == 1:
                new_chromosome1[idx] = 0

        index = random.sample(range(0, len(chromosome2)), int(len(chromosome2) * 0.1))
        for idx in index:
            if new_chromosome2[idx] == 0:
                new_chromosome2[idx] = 1
            elif new_chromosome2[idx] == 1:
                new_chromosome2[idx] = 0

        index = random.sample(range(0, len(chromosome3)), int(len(chromosome3) * 0.1))
        dump_number = len(self.DUMPS)
        for idx in index:
            new_chromosome3[idx] = random.randint(0, dump_number - 1)

        return new_chromosome1, new_chromosome2, new_chromosome3

    def operator_3(self, chromosome1, chromosome2, chromosome3, swapper_arrivaltime=0, battery_car_time=0):
        a, b = sorted(random.sample(range(0, len(chromosome1)), 2))
        new_chromosome1 = chromosome1[:a] + chromosome1[a:b][::-1] + chromosome1[b:]

        a, b = sorted(random.sample(range(0, len(chromosome2)), 2))
        new_chromosome2 = chromosome2[:a] + chromosome2[a:b][::-1] + chromosome2[b:]

        a, b = sorted(random.sample(range(0, len(chromosome3)), 2))
        new_chromosome3 = chromosome3[:a] + chromosome3[a:b][::-1] + chromosome3[b:]

        return new_chromosome1, new_chromosome2, new_chromosome3

    def operator_4(self, chromosome1, chromosome2, chromosome3, swapper_arrivaltime, battery_car_time):
        copy_chromosome1=copy.deepcopy(chromosome1)

        chromosome1_individual = np.array(
            [copy_chromosome1[i:i + self.chromosome1_length] for i in range(0, len(copy_chromosome1), self.chromosome1_length)])
        a, b = np.where(chromosome1_individual == 0)

        repaired_b = []
        for shovel, truck in self.SHOVEL_TRUCK.items():

            seen = set()

            for id in truck:
                trip = b[a == id]
                for index in range(len(swapper_arrivaltime[id])):
                    num = trip[index]
                    subtract = True
                    while (num in seen) | (num == 0):
                        if (num < index) & ((num - 1) not in seen) & (num > 1):
                            num -= 1
                        elif (subtract == False) & (num + 1 not in seen) & (num + 1 <= self.chromosome1_length):
                            num += 1
                        else:
                            if subtract:
                                num -= 1
                                if num == 0:
                                    subtract = False
                            else:
                                num += 1
                                if num == trip[index]:
                                    break

                    repaired_b.append(num)
                    seen.add(num)

                for index in range(len(swapper_arrivaltime[id]), len(trip)):
                    repaired_b.append(trip[index])

        repaired_b = np.array(repaired_b)
        repaired_chromosome1_individual = np.ones(chromosome1_individual.shape, dtype=int)

        repaired_chromosome1_individual[a, repaired_b] = 0

        repaired_chromosome1_individual = np.reshape(repaired_chromosome1_individual, -1)

        return list(repaired_chromosome1_individual), chromosome2, chromosome3

    def operator_5(self, chromosome1, chromosome2, chromosome3, swapper_arrivaltime, battery_car_time):
        chromosome3_individual = np.array([chromosome3[i:i + self.chromosome3_length] for i in range(0, len(chromosome3), self.chromosome3_length)])

        # valid_times = [max(i) for i in swapper_arrivaltime if i]
        # length = math.ceil(max(valid_times) + 1)
        time_list=[]
        for i in swapper_arrivaltime:
            time_list+=i
        length = math.ceil(max(time_list) + 1)
        arrive_time_array = np.zeros((len(self.DUMPS), length), dtype=int)
        truck_trip_dict = {}

        for i in range(len(swapper_arrivaltime)):
            time_list = swapper_arrivaltime[i]
            for j in range(len(time_list)):
                swapper_station = chromosome3_individual[i, j]
                arrive_time = time_list[j]

                arrive_time_array[int(swapper_station), math.ceil(arrive_time)] += 1


                if f"{swapper_station, math.ceil(arrive_time)}" not in truck_trip_dict:
                    truck_trip_dict[f"{int(swapper_station), math.ceil(arrive_time)}"] = []
                    truck_trip_dict[f"{int(swapper_station), math.ceil(arrive_time)}"].append((i, j))
                else:
                    truck_trip_dict[f"{int(swapper_station), math.ceil(arrive_time)}"].append((i, j))

        j = 0
        while j <= arrive_time_array.shape[1] - SWAPPING_TIME:
            if len(arrive_time_array[:, j][arrive_time_array[:, j] > 0]) != 0:
                scanning_area = arrive_time_array[:, j:j + SWAPPING_TIME]

                while True:
                    truck_number = np.sum(scanning_area, axis=1)
                    if len(truck_number[truck_number > 1]) == 0:
                        break
                    else:
                        min_number = np.min(truck_number)
                        max_number = np.max(truck_number)
                        if (max_number - min_number) <= 1:
                            break
                        else:
                            move_swapper = np.where(truck_number == max_number)[0][0]
                            target_swapper = np.where(truck_number == min_number)[0][0]

                            move_row = scanning_area[move_swapper]

                            if max(move_row) > 1:
                                truck_trip_index = np.where(move_row == max(move_row))[0][0]

                            else:
                                truck_trip_index = np.where(move_row == 1)[0][1]

                            chang_index = truck_trip_dict[f'{(int(move_swapper), int(j + truck_trip_index))}'][-1]
                            chromosome3_individual[chang_index[0]][chang_index[1]] = target_swapper

                            scanning_area[move_swapper, truck_trip_index] -= 1
                            scanning_area[target_swapper, truck_trip_index] += 1

                j = j + SWAPPING_TIME + 1

            else:
                j += 1

        repaired_chromosome3 = np.reshape(chromosome3_individual, -1)

        return chromosome1, chromosome2, list(repaired_chromosome3)

    def operator_6(self, chromosome1, chromosome2, chromosome3, swapper_arrivaltime, battery_car_time):
        chromosome2_individual = np.array(
            [chromosome2[i:i + self.chromosome2_length] for i in range(0, len(chromosome2), self.chromosome2_length)])

        for i in range(len(battery_car_time)):
            time_list = battery_car_time[i]
            for j in time_list:
                index = math.ceil(j)
                if index < self.chromosome2_length:
                    chromosome2_individual[i][index] = 1

        repaired_chromosome2 = np.reshape(chromosome2_individual, -1)

        return chromosome1, list(repaired_chromosome2), chromosome3

    def evaluate_wrapper(self, chromosome1, chromosome2, chromosome3):
        truck_repaired_schedule, car_repaired_schedule, swapper_arrivaltime, battery_car_time, fitness_value1, fitness_value2, fitness_value1_1, fitness_value1_2, fitness_value1_3,fitness_value2_1,fitness_value2_2  ,total_energy= EMineSim.Simulation(
            chromosome1, chromosome2, chromosome3, self.chromosome1_length, self.chromosome2_length, False)

        return truck_repaired_schedule, car_repaired_schedule, swapper_arrivaltime, battery_car_time, fitness_value1, fitness_value2, fitness_value1_1, fitness_value1_2, fitness_value1_3,fitness_value2_1,fitness_value2_2 ,total_energy

    def multiprocess_evaluate(self, population1, population2, population3):
        n=len(population1)
        with multiprocessing.Pool(processes=processes_number) as pool:
            result = pool.starmap(self.evaluate_wrapper ,[(population1[i], population2[i], population3[i]) for i in range(n)])

        truck_repaired_schedule, car_repaired_schedule, swapper_arrivaltime, battery_car_time, fitness_value1, fitness_value2, fitness_value1_1, fitness_value1_2, fitness_value1_3,fitness_value2_1,fitness_value2_2 ,total_energy= zip(
            *result)

        return list(truck_repaired_schedule), list(car_repaired_schedule), list(swapper_arrivaltime), list(
            battery_car_time), fitness_value1, fitness_value2, fitness_value1_1, fitness_value1_2, fitness_value1_3,fitness_value2_1,fitness_value2_2 ,total_energy

    def fast_non_dominated_sort(self, values1, values2):
        S = [[] for i in range(0, len(values1))]
        front = [[]]
        n = [0 for i in range(0, len(values1))]
        rank = [0 for i in range(0, len(values1))]

        for p in range(0, len(values1)):
            S[p] = []
            n[p] = 0
            for q in range(0, len(values1)):
                if (values1[p] >= values1[q] and values2[p] >= values2[q]) or (
                        values1[p] >= values1[q] and values2[p] > values2[q]) or (
                        values1[p] > values1[q] and values2[p] >= values2[q]):
                    if q not in S[p]:
                        S[p].append(q)
                elif (values1[q] > values1[p] and values2[q] > values2[p]) or (
                        values1[q] >= values1[p] and values2[q] > values2[p]) or (
                        values1[q] > values1[p] and values2[q] >= values2[p]):
                    n[p] = n[p] + 1
            if n[p] == 0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)

        i = 0
        while front[i] != []:
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)
            i = i + 1
            front.append(Q)

        del front[len(front) - 1]
        return front

    def crowding_distance(self, values1, values2, non_dominated_sorted):
        crowding_distance_values = []
        for i in range(len(non_dominated_sorted)):
            front = non_dominated_sorted[i]

            sorted1 = sorted(front, key=lambda i: values1[i], reverse=MAXIMIZED)
            sorted2 = sorted(front, key=lambda i: values2[i], reverse=MAXIMIZED)

            distance = [0 for i in range(0, len(front))]
            distance[0] = 4444444444444444
            distance[len(front) - 1] = 4444444444444444

            for k in range(1, len(front) - 1):
                if (max(values1) - min(values1)) != 0:
                    distance[k] = distance[k] + abs(values1[sorted1[k + 1]] - values1[sorted1[k - 1]]) / (
                            max(values1) - min(values1))
                else:
                    distance[k] += 0
            for k in range(1, len(front) - 1):
                if (max(values2) - min(values2)) != 0:
                    distance[k] = distance[k] + abs(values2[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (
                            max(values2) - min(values2))
                else:
                    distance[k] += 0
            crowding_distance_values.append(distance)
        return crowding_distance_values

    def is_better(self, current_fitness_value1, current_fitness_value2, new_fitness_value1, new_fitness_value2):
        return (max(current_fitness_value1) < max(new_fitness_value1)) & (
                max(current_fitness_value2) < max(new_fitness_value2))

    def NSPLNS_evolution(self):
        global cur_best_goal_list
        global cur_best_schedule_list
        cur_best_goal_list = []
        cur_best_schedule_list = []

        result = {"Objective1_values": [], "Objective1_1_values": [], "Objective1_2_values": [],
                  "Objective1_3_values": [], "Objective2_values": [], "Objective2_1values": [],"Objective2_2values": [],"Pareto_fronts": [], "Crowding_distance": [],"total_energy":[]}

        operator_list = [self.operator_1, self.operator_2, self.operator_3, self.operator_4, self.operator_5,
                         self.operator_6]

        j = 0
        for iteration in range(self.generation):
            time1=time.time()
            crossover_population1 = self.CHROMOSOME1_POPULATION[:]
            crossover_population2 = self.CHROMOSOME2_POPULATION[:]
            crossover_population3 = self.CHROMOSOME3_POPULATION[:]


            crossover1 = []
            crossover2 = []
            crossover3 = []

            timehere = time.time()
            length=len(crossover_population1)
            for o in range(3):
                operator = operator_list[o]
                for j in range(length):
                    crossover1_1,crossover2_1,crossover3_1=operator(crossover_population1[j],crossover_population2[j],crossover_population3[j])
                    crossover1.append(crossover1_1)
                    crossover2.append(crossover2_1)
                    crossover3.append(crossover3_1)

            crossover_population1.extend(crossover1)
            crossover_population2.extend(crossover2)
            crossover_population3.extend(crossover3)

            for o in range(3,6):
                operator = operator_list[o]
                crossover1, crossover2, swapper_arrivaltime, battery_car_time, fitness_value1, fitness_value2, fitness_value1_1, fitness_value1_2, fitness_value1_3,fitness_value2_1,fitness_value2_2 ,total_energy= self.multiprocess_evaluate(crossover1, crossover2, crossover3)

                timehere = time.time()
                length = len(crossover1)
                improved_crossover1 = []
                improved_crossover2 = []
                improved_crossover3 = []
                for j in range(length):
                    crossover1_1,crossover2_1,crossover3_1=operator(crossover1[j],crossover2[j],crossover3[j],swapper_arrivaltime[j],battery_car_time[j])
                    improved_crossover1.append(crossover1_1)
                    improved_crossover2.append(crossover2_1)
                    improved_crossover3.append(crossover3_1)
                crossover1 = copy.deepcopy(improved_crossover1)
                crossover2 = copy.deepcopy(improved_crossover2)
                crossover3 = copy.deepcopy(improved_crossover3)

            crossover_population1.extend(crossover1)
            crossover_population2.extend(crossover2)
            crossover_population3.extend(crossover3)

            crossover_population1, crossover_population2, swapper_arrivaltime, battery_car_time, fitness_value1, fitness_value2, fitness_value1_1, fitness_value1_2, fitness_value1_3,fitness_value2_1,fitness_value2_2,total_energy = self.multiprocess_evaluate(crossover_population1, crossover_population2, crossover_population3)

            non_dominated_sorted = self.fast_non_dominated_sort(fitness_value1, fitness_value2)
            crowding_distance_values = self.crowding_distance(fitness_value1, fitness_value2, non_dominated_sorted)

            if iteration==(self.generation-1):
                for i in non_dominated_sorted[0]:
                    EMineSim.Simulation(
                        crossover_population1[i], crossover_population2[i], crossover_population3[i], self.chromosome1_length,
                        self.chromosome2_length, True,self.algorithm_name,self.seed)

            f1 = np.array(fitness_value1)[non_dominated_sorted[0]]
            f2 = np.array(fitness_value2)[non_dominated_sorted[0]]

            print(f"iteration{iteration+1},time:{time.time() - time1}s,obj1_values:{f1},obj2_values:{f2}")
            fitness_value1_pareto = np.array(fitness_value1)[non_dominated_sorted[0]]
            fitness_value1_1pareto = np.array(fitness_value1_1)[non_dominated_sorted[0]]
            fitness_value1_2pareto = np.array(fitness_value1_2)[non_dominated_sorted[0]]
            fitness_value1_3pareto = np.array(fitness_value1_3)[non_dominated_sorted[0]]
            fitness_value2_pareto = np.array(fitness_value2)[non_dominated_sorted[0]]
            fitness_value2_1pareto = np.array(fitness_value2_1)[non_dominated_sorted[0]]
            fitness_value2_2pareto = np.array(fitness_value2_2)[non_dominated_sorted[0]]
            total_energypareto = np.array(total_energy)[non_dominated_sorted[0]]
            result["Objective1_values"].append([int(value) for value in fitness_value1_pareto])
            result["Objective1_1_values"].append([int(value) for value in fitness_value1_1pareto])
            result["Objective1_2_values"].append([int(value) for value in fitness_value1_2pareto])
            result["Objective1_3_values"].append([int(value) for value in fitness_value1_3pareto])
            result["Objective2_values"].append([int(value) for value in fitness_value2_pareto])
            result["Objective2_1values"].append([int(value) for value in fitness_value2_1pareto])
            result["Objective2_2values"].append([int(value) for value in fitness_value2_2pareto])
            result["Pareto_fronts"].append(non_dominated_sorted[0])
            result["Crowding_distance"].append(crowding_distance_values[0])
            result["total_energy"].append([int(value) for value in total_energypareto])

            new_solution_index = []
            for i in range(0, len(non_dominated_sorted)):
                front = non_dominated_sorted[i]
                if len(new_solution_index) == self.population_size:
                    break
                if len(new_solution_index) + len(front) <= self.population_size:
                    new_solution_index.extend(front)
                else:
                    distance = crowding_distance_values[i]
                    sorted_index = np.argsort(distance)[::-1]
                    for x in sorted_index:
                        new_solution_index.append(front[x])
                        if len(new_solution_index) == self.population_size:
                            break

            self.CHROMOSOME1_POPULATION = copy.deepcopy([crossover_population1[i] for i in new_solution_index])
            self.CHROMOSOME2_POPULATION = copy.deepcopy([crossover_population2[i] for i in new_solution_index])
            self.CHROMOSOME3_POPULATION = copy.deepcopy([crossover_population3[i] for i in new_solution_index])

            gc.collect()

        return result
