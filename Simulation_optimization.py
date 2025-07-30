import simpy
import config
import time
import pandas as pd

T, payload = config.get_para("T&p")
dumps, shovels, truck_num = config.get_para("object")
LOADING_TIME, UNLOADING_TIME = config.get_para("time")
path_length = config.get_para("path")
shovel_truck, shovel_dump = config.get_para("match")
swapping_stations = ['U1','U2','U3']


MAX_NUMBERS_OF_SWAPPING_BATTERY = 3
SWAPPING_POS = 1
SWAPPING_TIME = 12
maximum_SOC=50

BATTERY_TRAVELING_TIME = 15
BOARD_INTERVAL = config.get_para("board_interval")
MINIMUM_INTERVAL = 60

charging_stations = ["Charger 1"]
BATTERY_CHARGING_NUMBER = 40
GET_BATTERY_NUMBER = 1
BATTERYGET_TIME = 0


initial_SoC = 95
lower_SoC = 20
upper_SoC = 95
CHARGING_SPEED = 1
BATTERY_CAR_CAPACITY = 2

BATTERY_CAR_DECISION = True
WAITING_OPTIMIZATION = True

consumption_coefficient = 1


class Truck(object):
    def __init__(self, env, truck_id):
        self.truck_id = truck_id

        self.binding_shovel = ''

        self.travel_time = 0

        self.SoC = 0

        self.output_time = []
        self.output_space = []
        self.output_task = []

        self.output_SoC_time = []
        self.output_SoC = []

        self.swapper_arrivaltime = []

        self.schedule = []
        self.repaired_schedule = []
        self.trip_index = 0

        self.swapper_schedule = []
        self.swapper_repaired_schedule = []
        self.swapper_trip = 0
        self.swapper_trip_true = 0

        self.departure_time = 0
        self.arrival_time = 0

        self.env = env
        self.trip_number = 0


    def position_update(self, e_q, u_q, s_q, next_position, next_position_id, next_time, next_consumption):
        global truck_list

        yield self.env.timeout(next_time[0])
        truck_list[self.truck_id].travel_time = next_time
        self.SoC -= next_consumption

        if next_position == 'Shovel':
            self.env.process(walk_process(self.env, '-', next_position_id, self.truck_id, e_q, "go to excavator"))
        elif next_position == 'Dump':
            self.env.process(walk_process(self.env, '-', next_position_id, self.truck_id, u_q, "go to unload_point"))
        elif next_position == 'Swapper':
            self.env.process(walk_process(self.env, '-', next_position_id, self.truck_id, s_q, "go to swapper"))


def walk_process(env, start_area, goal_area, truck_id, next_q, direction):
    global truck_list
    global waiting_truck_number
    global total_energy
    while True:
        if direction == "go to excavator":
            walk_time = truck_list[truck_id].travel_time[0]

            truck_list[truck_id].output_space.append(start_area)
            truck_list[truck_id].output_time.append(round(env.now, 2))
            truck_list[truck_id].output_task.append("Start Moving")
            truck_list[truck_id].output_SoC.append(round(truck_list[truck_id].SoC, 2))

            yield env.timeout(float(walk_time))
            truck_list[truck_id].SoC -= round(consumption_coefficient * walk_time, 2)
            total_energy += consumption_coefficient * walk_time

            next_q[goal_area].put(truck_id)
            truck_list[truck_id].arrival_time = round(env.now, 2)

        elif direction == "go to unload_point":
            walk_time = truck_list[truck_id].travel_time[0]

            truck_list[truck_id].output_space.append(start_area)
            truck_list[truck_id].output_time.append(round(env.now, 2))
            truck_list[truck_id].output_task.append("Start Moving")
            truck_list[truck_id].output_SoC.append(round(truck_list[truck_id].SoC, 2))

            yield env.timeout(float(walk_time))
            truck_list[truck_id].SoC -= round(consumption_coefficient * walk_time, 2)
            total_energy += consumption_coefficient * walk_time

            next_q[goal_area].put(truck_id)
            truck_list[truck_id].arrival_time = round(env.now, 2)

        else:
            walk_time = path_length[(start_area, goal_area)][0]

            truck_list[truck_id].output_space.append(start_area)
            truck_list[truck_id].output_time.append(round(env.now, 2))
            truck_list[truck_id].output_task.append("Start Moving")
            truck_list[truck_id].output_SoC.append(round(truck_list[truck_id].SoC, 2))

            yield env.timeout(float(walk_time))
            truck_list[truck_id].SoC -= round(consumption_coefficient * walk_time, 2)
            total_energy += consumption_coefficient * walk_time

            next_q[goal_area].put(truck_id)

            truck_list[truck_id].arrival_time = round(env.now, 2)

        return

class Battery_car(object):
    def __init__(self, env, battery_car_id, capacity):

        self.env = env
        self.battery_car_id = battery_car_id
        self.capacity = capacity

        self.car_schedule = []
        self.car_repaired_schedule = []
        self.interval = BOARD_INTERVAL
        self.minimum_interval = MINIMUM_INTERVAL

        self.trip_index = 0

        self.output_time = []
        self.output_space = []
        self.output_task = []
        self.trip_number = 0

    def battery_dropoff_pickup_process(self, battery_queue, charging_station_id, charging_ua_b, charging_a_b,
                                       charging_battery_resource, swapping_station_a_b, battery_car_id):
        global battery_car_return

        self.output_time.append(round(self.env.now, 2))
        self.output_space.append(battery_car_id)
        self.output_task.append('Start Moving')

        yield self.env.timeout(BATTERY_TRAVELING_TIME)

        self.output_time.append(round(self.env.now, 2))
        self.output_space.append(charging_station_id)
        self.output_task.append('Arriving Charging Station')

        with charging_battery_resource[charging_station_id].request() as req:
            yield req

            self.output_time.append(round(self.env.now, 2))
            self.output_space.append(charging_station_id)
            self.output_task.append('Starting Loading')


            for i in battery_queue:
                charging_ua_b[charging_station_id].put(i)
            yield self.env.timeout(BATTERYGET_TIME)


            battery_number = 0
            battery_level_list = []
            while battery_number < self.capacity:
                battery_level = yield charging_a_b[charging_station_id].get()
                battery_level_list.append(battery_level)
                battery_number += 1
            yield self.env.timeout(BATTERYGET_TIME)

            self.output_time.append(round(self.env.now, 2))
            self.output_space.append(charging_station_id)
            self.output_task.append('End Loading')

        self.output_time.append(round(self.env.now, 2))
        self.output_space.append(charging_station_id)
        self.output_task.append('Starting Moving')

        yield self.env.timeout(BATTERY_TRAVELING_TIME)

        self.output_time.append(round(self.env.now, 2))
        self.output_space.append(battery_car_id)
        self.output_task.append('Arrival Swapping Station')

        battery_car_return[battery_car_id] = True

        for i in battery_level_list:
            yield swapping_station_a_b[battery_car_id].put(i)

    def Car_decision(self, swapping_station_a_b, swapping_station_ua_b, charging_a_b, charging_ua_b):
        global battery_car_return

        if BATTERY_CAR_DECISION == False:
            while True:
                if len(swapping_station_ua_b[self.battery_car_id]) >= self.capacity:
                    ua_b = []
                    count = 0
                    while count < self.capacity:
                        ua_b.append(swapping_station_ua_b[self.battery_car_id][0])
                        del swapping_station_ua_b[self.battery_car_id][0]
                        count += 1

                    self.env.process(
                        self.battery_dropoff_pickup_process(ua_b, charging_stations[0], charging_ua_b, charging_a_b,
                                                            charging_battery_resource, swapping_station_a_b,
                                                            self.battery_car_id))
                    yield self.env.timeout(60)

                else:
                    yield self.env.timeout(60)

        if BATTERY_CAR_DECISION == True:
            sum_interval = 0
            while len(self.car_schedule) != 0:
                if len(swapping_station_ua_b[self.battery_car_id]) >= self.capacity:
                    if sum_interval >= self.minimum_interval:
                        ua_b = []
                        count = 0
                        while count < self.capacity:
                            ua_b.append(swapping_station_ua_b[self.battery_car_id][0])
                            del swapping_station_ua_b[self.battery_car_id][0]
                            count += 1

                        self.env.process(
                            self.battery_dropoff_pickup_process(ua_b, charging_stations[0], charging_ua_b,
                                                                charging_a_b,
                                                                charging_battery_resource, swapping_station_a_b,
                                                                self.battery_car_id))

                        del self.car_schedule[0]
                        self.car_repaired_schedule[self.trip_index] = 1
                        self.trip_index += 1
                        self.trip_number += 1
                        battery_car_return[self.battery_car_id] = False
                        sum_interval = 0

                    else:
                        del self.car_schedule[0]
                        self.car_repaired_schedule[self.trip_index] = 0
                        self.trip_index += 1
                        sum_interval += self.interval
                        yield self.env.timeout(self.interval)

                else:
                    del self.car_schedule[0]
                    self.car_repaired_schedule[self.trip_index] = 0
                    self.trip_index += 1
                    sum_interval += self.interval
                    yield self.env.timeout(self.interval)


class Battery_Charging_station(object):
    def __init__(self, charging_station_id, env):
        self.charging_station_id = charging_station_id
        self.env = env
        self.a_b = simpy.Store(self.env)
        self.ua_b = simpy.Store(self.env)
        self.resource = simpy.Resource(self.env, capacity=BATTERY_CHARGING_NUMBER)
        self.battery_get_resource = simpy.Resource(self.env, capacity=GET_BATTERY_NUMBER)

    def charging_process(self, remaining_Soc):
        global upper_SoC
        global CHARGING_SPEED

        with self.resource.request() as req:
            yield req
            charging_time = (upper_SoC - remaining_Soc) / CHARGING_SPEED
            yield self.env.timeout(charging_time)
            self.a_b.put(upper_SoC)

    def charging_location(self, ):
        while True:
            remaining_Soc = yield self.ua_b.get()
            self.env.process(self.charging_process(remaining_Soc))

class Battery_swapping_station(object):
    global truck_list
    global upper_SoC
    global charging_power
    global total_waiting_time

    def __init__(self, charger_id, env):
        self.swapper_id = charger_id
        self.env = env
        self.s_q = simpy.Store(self.env)
        self.resource = simpy.Resource(self.env, capacity=SWAPPING_POS)
        self.a_b = simpy.Store(self.env)
        self.ua_b = []

        self.batterycar_time = []

    def swapping_process(self, truck_id, e_q, charging_a_b, charging_ua_b):
        global truck_list
        global upper_SoC
        global charging_power
        global total_swapping_time
        global battery_car_list
        global waiting_truck_number

        truck_list[truck_id].arrival_time = self.env.now
        waiting_truck_number[self.swapper_id] += 1

        truck_list[truck_id].output_space.append(self.swapper_id)
        truck_list[truck_id].output_time.append(round(self.env.now, 2))
        truck_list[truck_id].output_task.append("Arrival Swapping Station")
        truck_list[truck_id].output_SoC.append(round(truck_list[truck_id].SoC, 2))

        truck_list[truck_id].swapper_arrivaltime.append(round(self.env.now, 2))

        with self.resource.request() as req:
            yield req

            yield self.a_b.get()

            truck_list[truck_id].output_space.append(self.swapper_id)
            truck_list[truck_id].output_time.append(round(self.env.now, 2))
            truck_list[truck_id].output_task.append("Start Swapping")
            truck_list[truck_id].output_SoC.append(round(truck_list[truck_id].SoC, 2))

            total_swapping_time += round(self.env.now - truck_list[truck_id].arrival_time, 2)
            yield self.env.timeout(SWAPPING_TIME)
            truck_list[truck_id].swapper_trip_true += 1

            self.ua_b.append(round(truck_list[truck_id].SoC, 2))

            if len(self.ua_b) == battery_car_list[self.swapper_id].capacity:
                self.batterycar_time.append(self.env.now)

            truck_list[truck_id].SoC = upper_SoC
            goal_area = truck_list[truck_id].binding_shovel
            goal_area = shovel_dump[goal_area]

            truck_list[truck_id].travel_time = path_length[self.swapper_id, goal_area]
            truck_list[truck_id].departure_time = round(self.env.now, 2)

            truck_list[truck_id].output_space.append(self.swapper_id)
            truck_list[truck_id].output_time.append(round(self.env.now, 2))
            truck_list[truck_id].output_task.append("End Swapping")
            truck_list[truck_id].output_SoC.append(round(truck_list[truck_id].SoC, 2))

            self.env.process(walk_process(self.env, self.swapper_id, goal_area, truck_id, u_q, "go to unload_point"))

        waiting_truck_number[self.swapper_id] -= 1

    def swapping_location(self, e_q):
        while True:
            truck_id = yield self.s_q.get()
            self.env.process(self.swapping_process(truck_id, e_q, charging_a_b, charging_ua_b))


class Excavator(object):
    def __init__(self, excavator_id, env):
        self.excavator_id = excavator_id
        self.env = env
        self.e_q = simpy.Store(self.env)
        self.resource = simpy.Resource(self.env, capacity=1)

    def loading_process(self, truck_id, u_q):
        global truck_list
        global real_shovel_workload
        global total_shovel_waiting_time
        global payload

        truck_list[truck_id].arrival_time = self.env.now

        truck_list[truck_id].output_space.append(self.excavator_id)
        truck_list[truck_id].output_time.append(round(self.env.now, 2))
        truck_list[truck_id].output_task.append("Arrival Excavator")
        truck_list[truck_id].output_SoC.append(round(truck_list[truck_id].SoC, 2))

        with self.resource.request() as req:
            yield req

            truck_list[truck_id].output_space.append(self.excavator_id)
            truck_list[truck_id].output_time.append(round(self.env.now, 2))
            truck_list[truck_id].output_task.append("Start Loading")
            truck_list[truck_id].output_SoC.append(round(truck_list[truck_id].SoC, 2))

            truck_list[truck_id].departure_time = self.env.now
            total_shovel_waiting_time += truck_list[truck_id].departure_time - truck_list[truck_id].arrival_time

            load_time = LOADING_TIME[self.excavator_id]
            yield self.env.timeout(float(load_time))

            truck_list[truck_id].output_space.append(self.excavator_id)
            truck_list[truck_id].output_time.append(round(self.env.now, 2))
            truck_list[truck_id].output_task.append("End Loading")
            truck_list[truck_id].output_SoC.append(round(truck_list[truck_id].SoC, 2))

            real_shovel_workload[self.excavator_id] += payload[truck_id]

            goal_area = shovel_dump[self.excavator_id]

            truck_list[truck_id].travel_time = path_length[self.excavator_id, goal_area]
            truck_list[truck_id].trip_number += 1

            self.env.process(walk_process(self.env, self.excavator_id, goal_area, truck_id, u_q, "go to unload_point"))

    def loading_location(self, u_q):
        while True:
            truck_id = yield self.e_q.get()
            self.env.process(self.loading_process(truck_id, u_q))


class Dump(object):
    def __init__(self, dump_id, env):

        self.dump_id = dump_id
        self.env = env
        self.u_q = simpy.Store(self.env)
        self.resource = simpy.Resource(self.env, capacity=1)
        self.arrival_time = 0
        self.departure_time = 0

    def unloading_process(self, truck_id, e_q, s_q):
        global truck_list
        global payload
        global lower_SoC
        global total_dump_waiting_time

        truck_list[truck_id].arrival_time = self.env.now

        truck_list[truck_id].output_space.append(self.dump_id)
        truck_list[truck_id].output_time.append(round(self.env.now, 2))
        truck_list[truck_id].output_task.append("Arrival Dump")
        truck_list[truck_id].output_SoC.append(round(truck_list[truck_id].SoC, 2))

        with self.resource.request() as req:
            yield req
            truck_list[truck_id].output_space.append(self.dump_id)
            truck_list[truck_id].output_time.append(round(self.env.now, 2))
            truck_list[truck_id].output_task.append("Start Unloading")
            truck_list[truck_id].output_SoC.append(round(truck_list[truck_id].SoC, 2))

            truck_list[truck_id].departure_time = self.env.now
            total_dump_waiting_time += truck_list[truck_id].departure_time - truck_list[truck_id].arrival_time

            unload_time = UNLOADING_TIME[self.dump_id]
            yield self.env.timeout(float(unload_time))

            real_dump_workload[self.dump_id] += payload[truck_id]

            truck_list[truck_id].output_space.append(self.dump_id)
            truck_list[truck_id].output_time.append(round(self.env.now, 2))
            truck_list[truck_id].output_task.append("End Unloading")
            truck_list[truck_id].output_SoC.append(round(truck_list[truck_id].SoC, 2))

            if len(truck_list[truck_id].schedule) != 0:
                goal_area = truck_list[truck_id].binding_shovel

                swapper_index = truck_list[truck_id].swapper_schedule[0]
                walk_time = path_length[(self.dump_id, dumps[swapper_index])][0]

                Electricity_consumption_of_a_tour = round(consumption_coefficient * (path_length[self.dump_id, goal_area][0] + path_length[goal_area, self.dump_id][0] + walk_time), 2)

                if truck_list[truck_id].SoC >= maximum_SOC:
                    truck_list[truck_id].schedule[0] = 1

                if (truck_list[truck_id].SoC <= lower_SoC) or (
                        truck_list[truck_id].SoC - Electricity_consumption_of_a_tour <= 10):
                    truck_list[truck_id].schedule[0] = 0

                if truck_list[truck_id].schedule[0] == 0:

                    del truck_list[truck_id].schedule[0]
                    truck_list[truck_id].repaired_schedule[truck_list[truck_id].trip_index] = 0
                    truck_list[truck_id].trip_index += 1

                    swapper_index = truck_list[truck_id].swapper_schedule[0]

                    goal_area = swapping_stations[swapper_index]

                    del truck_list[truck_id].swapper_schedule[0]
                    truck_list[truck_id].swapper_repaired_schedule[truck_list[truck_id].swapper_trip] = swapper_index
                    truck_list[truck_id].swapper_trip += 1

                    self.env.process(walk_process(self.env, self.dump_id, goal_area, truck_id, s_q, "go to swapper"))


                elif truck_list[truck_id].schedule[0] == 1:

                    del truck_list[truck_id].schedule[0]
                    truck_list[truck_id].repaired_schedule[truck_list[truck_id].trip_index] = 1
                    truck_list[truck_id].trip_index += 1

                    goal_area = truck_list[truck_id].binding_shovel
                    truck_list[truck_id].travel_time = path_length[self.dump_id, goal_area]

                    if self.env.now + path_length[self.dump_id, goal_area][0] + path_length[goal_area, self.dump_id][
                        0] + LOADING_TIME[goal_area] <= T:
                        self.env.process(
                            walk_process(self.env, self.dump_id, goal_area, truck_id, e_q, "go to excavator"))

    def unloading_location(self, e_q, s_q):
        global Electricity_consumption_of_a_tour
        global truck_list
        while True:
            truck_id = yield self.u_q.get()
            if truck_list[truck_id].output_task[-2] == 'End Loading':
                self.env.process(self.unloading_process(truck_id, e_q, s_q))
            if truck_list[truck_id].output_task[-2] == 'End Swapping':
                goal_area = truck_list[truck_id].binding_shovel
                if self.env.now + path_length[self.dump_id, goal_area][0] + path_length[goal_area, self.dump_id][0] + \
                        LOADING_TIME[goal_area] <= T:
                    self.env.process(walk_process(self.env, self.dump_id, goal_area, truck_id, e_q, "go to excavator"))


def find_key(element, data):
    for key, values in data.items():
        if element in values:
            return key
    return None


def Simulation(chromosome1, chromosome2, chromosome3, length1, length2, output, algorithm_name=None, seed=None):
    global excavator_site_list
    global dump_site_list
    global charger_site_list
    global swapping_site_list
    global battery_car_list

    global e_q
    global u_q
    global s_q

    global charging_a_b
    global charging_ua_b

    global truck_list
    global battery_car_list

    global real_shovel_workload
    global real_dump_workload

    global shovels
    global dumps
    global swapping_stations
    global waiting_truck_number
    global battery_car_return

    global total_shovel_waiting_time
    global total_dump_waiting_time
    global total_swapping_time
    global total_batteryget_time
    global charging_battery_resource
    global swapping_station_a_b

    global total_energy


    env = simpy.Environment()

    start_time = time.time()

    individual = [chromosome1[i:i + length1] for i in range(0, len(chromosome1), length1)]
    individual1 = [chromosome3[i:i + 10] for i in range(0, len(chromosome3), 10)]

    truck_list = {}

    for shovel, trucks in shovel_truck.items():
        for truck_id in trucks:
            truck_list[truck_id] = (Truck(env, truck_id))
            truck_list[truck_id].binding_shovel = shovel
            truck_list[truck_id].SoC = upper_SoC
            truck_list[truck_id].schedule = individual[truck_id].copy()
            truck_list[truck_id].repaired_schedule = individual[truck_id].copy()
            truck_list[truck_id].swapper_schedule = individual1[truck_id].copy()
            truck_list[truck_id].swapper_repaired_schedule = individual1[truck_id].copy()


    truck_state = {}
    for i in range(truck_num):
        shovel = truck_list[i].binding_shovel
        truck_state[i] = ['Shovel', shovel, path_length[shovel_dump[shovel], shovel], 0]


    individual = [chromosome2[i:i + length2] for i in range(0, len(chromosome2), length2)]


    battery_car_list = {}
    battery_car_return = {}
    for i in range(len(swapping_stations)):
        battery_car_list[swapping_stations[i]] = Battery_car(env, swapping_stations[i], BATTERY_CAR_CAPACITY)
        battery_car_list[swapping_stations[i]].car_schedule = individual[i].copy()
        battery_car_list[swapping_stations[i]].car_repaired_schedule = individual[i].copy()
        battery_car_return[swapping_stations[i]] = True


    e_q = {}
    excavator_site_list = {}
    for i in shovels:
        excavator_site_list[i] = Excavator(i, env)
        e_q[i] = excavator_site_list[i].e_q

    u_q = {}
    dump_site_list = {}
    for i in dumps:
        dump_site_list[i] = Dump(i, env)
        u_q[i] = dump_site_list[i].u_q

    s_q = {}
    swapping_station_a_b = {}
    swapping_station_ua_b = {}
    swapping_site_list = {}
    waiting_truck_number = {}
    for i in swapping_stations:
        swapping_site_list[i] = Battery_swapping_station(i, env)
        s_q[i] = swapping_site_list[i].s_q
        swapping_station_a_b[i] = swapping_site_list[i].a_b
        for j in range(MAX_NUMBERS_OF_SWAPPING_BATTERY):
            swapping_station_a_b[i].put(upper_SoC)
        swapping_station_ua_b[i] = swapping_site_list[i].ua_b
        waiting_truck_number[i] = 0


    charging_site_list = {}
    charging_a_b = {}
    charging_ua_b = {}
    charging_battery_resource = {}
    for i in charging_stations:
        charging_site_list[i] = Battery_Charging_station(i, env)
        charging_battery_resource[i] = charging_site_list[i].battery_get_resource

        for j in range(2):
            charging_site_list[i].a_b.put(upper_SoC)

        charging_a_b[i] = charging_site_list[i].a_b
        charging_ua_b[i] = charging_site_list[i].ua_b

    total_shovel_waiting_time = 0
    total_dump_waiting_time = 0
    total_swapping_time = 0
    total_batteryget_time = 0
    total_energy = 0

    real_shovel_workload = {}
    real_dump_workload = {}
    for i in dumps:
        real_dump_workload[i] = 0
    for i in shovels:
        real_shovel_workload[i] = 0


    for i in shovels:
        env.process(excavator_site_list[i].loading_location(u_q))

    for i in dumps:
        env.process(dump_site_list[i].unloading_location(e_q, s_q))

    for i in swapping_stations:
        env.process(swapping_site_list[i].swapping_location(e_q))
        env.process(
            battery_car_list[i].Car_decision(swapping_station_a_b, swapping_station_ua_b, charging_a_b, charging_ua_b))

    for i in charging_stations:
        env.process(charging_site_list[i].charging_location())

    for i in range(truck_num):
        next_position = truck_state[i][0]
        next_position_id = truck_state[i][1]
        next_time = truck_state[i][2]
        next_consumption = truck_state[i][3]
        env.process(
            truck_list[i].position_update(e_q, u_q, s_q, next_position, next_position_id, next_time, next_consumption))


    env.run()


    truck_trip_number = []
    truck_repaired_schedule = []
    for truck_id in range(truck_num):
        truck_repaired_schedule += truck_list[truck_id].repaired_schedule
        truck_trip_number.append(truck_list[truck_id].trip_number)

    car_trip_number = []
    car_repaired_schedule = []
    for id in swapping_stations:
        car_repaired_schedule += battery_car_list[id].car_repaired_schedule
        car_trip_number.append(battery_car_list[id].trip_number)

    total_production = 0
    for i in shovels:
        total_production += real_shovel_workload[i]

    total_swapping_work_time = 0
    for truck_id in range(truck_num):
        total_swapping_work_time += (truck_list[truck_id].swapper_trip_true * SWAPPING_TIME)

    fitness_value1_1 = 182 * total_production
    fitness_value1_2 = -100 * sum(truck_trip_number)
    fitness_value1_3 = -100 * sum(car_trip_number)
    fitness_value1 = 182 * total_production - 100 * sum(truck_trip_number) - 100 * sum(car_trip_number)
    fitness_value2 = -1 * (
                total_swapping_time + total_swapping_work_time + total_dump_waiting_time + total_shovel_waiting_time)
    fitness_value2_1 = -total_swapping_time
    fitness_value2_2 = -total_swapping_work_time

    swapper_arrivaltime = []
    for truck_id, item in truck_list.items():
        swapper_arrivaltime.append(item.swapper_arrivaltime)

    battery_car_time = []
    for swapper_id, item in swapping_site_list.items():
        battery_car_time.append(item.batterycar_time)

    if output:
        output_truck_id = []
        output_truck_time = []
        output_truck_space = []
        output_truck_task = []
        output_truck_SoC = []
        for truck_id, item in truck_list.items():
            output_truck_id += [truck_id] * len(item.output_time)
            output_truck_time += item.output_time
            output_truck_space += item.output_space
            output_truck_task += item.output_task
            output_truck_SoC += item.output_SoC

            df = {"矿卡id": output_truck_id, "时间": output_truck_time, "位置": output_truck_space,
                  "任务": output_truck_task, "SoC": output_truck_SoC}
            if algorithm_name is not None:
                pd.DataFrame(df).to_csv(
                    f"truck_output_{algorithm_name}_{seed}_{total_production}_{fitness_value2}_{total_energy}.csv",
                    index=False, encoding='utf-8_sig')
            else:
                pd.DataFrame(df).to_csv(f"truck_output.csv", index=False, encoding='utf-8_sig')

        output_car_id = []
        output_car_time = []
        output_car_space = []
        output_car_task = []
        for id, item in battery_car_list.items():
            output_car_id += [id] * len(item.output_time)
            output_car_time += item.output_time
            output_car_space += item.output_space
            output_car_task += item.output_task

            df = {"板车id": output_car_id, "时间": output_car_time, "位置": output_car_space, "任务": output_car_task}
            if algorithm_name is not None:
                pd.DataFrame(df).to_csv(
                    f"car_output_{algorithm_name}_{seed}_{total_production}_{fitness_value2}_{total_energy}.csv",
                    index=False, encoding='utf-8_sig')
            else:
                pd.DataFrame(df).to_csv("car_output.csv", index=False, encoding='utf-8_sig')

    return truck_repaired_schedule, car_repaired_schedule, swapper_arrivaltime, battery_car_time, fitness_value1, fitness_value2, fitness_value1_1, fitness_value1_2, fitness_value1_3, fitness_value2_1, fitness_value2_2, total_energy
