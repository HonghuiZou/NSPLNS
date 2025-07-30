import pandas as pd

T = 1 * 480

empty_speed = 500  # m/min
heavy_speed = 500  # m/min

###################################### node data ######################################
raw_node_data = pd.read_excel("input_node.xlsx")

dump_data = raw_node_data[raw_node_data["type"] == "dump"]
dump_data.reset_index(drop=True, inplace=True)
dump_target_mass = {}
for i in range(len(dump_data)):
    dump_target_mass[dump_data["description"][i]] = dump_data["target"][i]

shovel_data = raw_node_data[raw_node_data["type"] == "shovel"]
shovel_data.reset_index(drop=True, inplace=True)
shovel_target_mass = {}
for i in range(len(shovel_data)):
    shovel_target_mass[str(shovel_data["description"][i])] = shovel_data["target"][i]

###################################### path ######################################
path= {
    ('L1', 'U1'): [4000],
    ('L1', 'U2'): [5500],
    ('L1', 'U3'): [5500],
    ('L2', 'U1'): [2355],
    ('L2', 'U2'): [1455],
    ('L2', 'U3'): [1500],
    ('L3', 'U1'): [1350],
    ('L3', 'U2'): [3000],
    ('L3', 'U3'): [3000],
    ('L4', 'U1'): [4500],
    ('L4', 'U2'): [6450],
    ('L4', 'U3'): [6450],
    ('L5', 'U1'): [2000],
    ('L5', 'U2'): [1500],
    ('L5', 'U3'): [1650],
    ('U1', 'L1'): [3750],
    ('U1', 'L2'): [2450],
    ('U1', 'L3'): [1350],
    ('U1', 'L4'): [4500],
    ('U1', 'L5'): [2000],
    ('U2', 'L1'): [5755],
    ('U2', 'L2'): [1455],
    ('U2', 'L3'): [3350],
    ('U2', 'L4'): [6500],
    ('U2', 'L5'): [1500],
    ('U3', 'L1'): [5700],
    ('U3', 'L2'): [1800],
    ('U3', 'L3'): [3350],
    ('U3', 'L4'): [6500],
    ('U3', 'L5'): [1655],

    ('L1', 'L1'): [0],
    ('L1', 'L2'): [0],
    ('L1', 'L3'): [0],
    ('L1', 'L4'): [0],
    ('L1', 'L5'): [0],
    ('L2', 'L1'): [0],
    ('L2', 'L2'): [0],
    ('L2', 'L3'): [0],
    ('L2', 'L4'): [0],
    ('L2', 'L5'): [0],
    ('L3', 'L1'): [0],
    ('L3', 'L2'): [0],
    ('L3', 'L3'): [0],
    ('L3', 'L4'): [0],
    ('L3', 'L5'): [0],
    ('L4', 'L1'): [0],
    ('L4', 'L2'): [0],
    ('L4', 'L3'): [0],
    ('L4', 'L4'): [0],
    ('L4', 'L5'): [0],
    ('L5', 'L1'): [0],
    ('L5', 'L2'): [0],
    ('L5', 'L3'): [0],
    ('L5', 'L4'): [0],
    ('L5', 'L5'): [0],

    ('U1', 'U1'): [0],
    ('U1', 'U2'): [0],
    ('U1', 'U3'): [0],
    ('U2', 'U1'): [0],
    ('U2', 'U2'): [0],
    ('U2', 'U3'): [0],
    ('U3', 'U1'): [0],
    ('U3', 'U2'): [0],
    ('U3', 'U3'): [0],
}
path_length_dict= {}
for (start, end), [distance] in sorted(path.items()):
    if start.startswith('U') and end.startswith('L'):
        time = distance / empty_speed
        path_length_dict[(start, end)] = [time]
    elif start.startswith('L') and end.startswith('U'):
        time = distance / heavy_speed
        path_length_dict[(start, end)] = [time]
    else:
        path_length_dict [(start, end)] = [distance]
###################################### shovel to dump ######################################
link_raw_data = pd.read_excel("input_link.xlsx")
shovel_dump = {}
for i in range(len(link_raw_data)):
    shovel_dump[str(link_raw_data["from_node_id"][i])] = link_raw_data["to_node_id"][i]

###################################### truck data ######################################
raw_truck_data = pd.read_excel("input_truck.xlsx")
truck_num = len(raw_truck_data)
payload = {}

for i in range(truck_num):
    payload[i] = 200

###################################### other data ######################################
dump_id_list = dump_data["description"].tolist()
temp_shovel_id_list = shovel_data["description"].tolist()
shovel_id_list = [str(shovel) for shovel in temp_shovel_id_list]

loading_time = {}
unloading_time = {}

S_1 = ['L1', 'L2', 'L3', 'L4', 'L5']
for i in ['L1', 'L2']:
    loading_time[i] = 6.5
for i in ['L3']:
    loading_time[i] = 5
for i in ['L4', 'L5']:
    loading_time[i] = 4.5

Pou = ['U1', 'U2', 'U3']
for i in Pou:
    unloading_time[i] = 1

truck_list = []
truck_id = 0
shovel_truck = {}
for shovel_id in shovel_id_list:
    shovel_truck[shovel_id] = []
    temp_group_size = 0
    while temp_group_size < 5:
        shovel_truck[shovel_id].append(truck_id)
        temp_group_size += 1
        truck_id += 1


def get_para(name):
    if name == 'T&p':
        return T, payload
    elif name == 'object':
        return dump_id_list, shovel_id_list, truck_num
    elif name == 'time':
        return loading_time, unloading_time
    elif name == 'match':
        return shovel_truck, shovel_dump
    elif name == 'speed':
        return empty_speed, heavy_speed
    elif name == 'path':
        return path_length_dict
    elif name == 'board_interval':
        return 1
