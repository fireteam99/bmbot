import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_FACTORY = actions.FUNCTIONS.Build_Factory_screen.id
_BUILD_STARPORT = actions.FUNCTIONS.Build_Starport_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_TRAIN_HELLION = actions.FUNCTIONS.Train_Hellion_quick.id
_SCV_GATHER = actions.FUNCTIONS.Harvest_Gather_SCV_screen.id
_TRAIN_MEDIVAC = actions.FUNCTIONS.Train_Medivac_quick.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_FACTORY = 27
_TERRAN_STARPORT = 28
_TERRAN_REFINERY = 20
_NEUTRAL_MINERAL_FIELD = 341
_VESPENE_GAS = 342

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'bm_agent_data'
ACTION_MOVE_SCV_TO_GAS_GEYSER1 = 'movescvtogasgeyser1'
ACTION_MOVE_SCV_TO_GAS_GEYSER2 = 'movescvtogasgeyser2'
ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'
ACTION_BUILD_FACTORY = 'buildfactory'
ACTION_BUILD_HELLION = 'buildhellion'
ACTION_BUILD_STARPORT = 'buildstarport'
ACTION_BUILD_MEDIVAC = 'buildmedivac'
ACTION_BUILD_REACTOR = 'buildreactor'
ACTION_BUILD_TECHLAB = 'buildtechlab'
ACTION_BUILD_SCV = 'buildscv'

refinery1_built = False
refinery2_built = False
geyser1_cord = None
geyser2_cord = None
gas_cap = 3
gas_geyser1_workers = 1
gas_geyser2_workers = 1
scvs_built = 0
gas_built = 0
scv_made = False

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_BUILD_FACTORY,
    ACTION_BUILD_HELLION,
    ACTION_BUILD_STARPORT,
    ACTION_BUILD_MEDIVAC,
    ACTION_BUILD_SCV,
]
# refineries are 97px wide

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class BMAgent(base_agent.BaseAgent):
    def __init__(self):
        super(BMAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_action = None
        self.previous_state = None

        self.cc_y = None
        self.cc_x = None

        self.move_number = 0


        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    def step(self, obs):
        super(BMAgent, self).step(obs)

        # unit_type = obs.observation["screen"][_UNIT_TYPE]
        # cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        # if cc_y.any() and scv_made == False:
        #     print("want to build scvs")
        #     unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        #     target = [int(unit_x.mean()), int(unit_y.mean())]
        #     if obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and _TRAIN_SCV in \
        #             obs.observation["available_actions"]:
        #         print("building scvs")
        #         return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])
        #     scv_made = True
        #     return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

        unit_type = obs.observation["screen"][_UNIT_TYPE]
        r_y, r_x = (unit_type == _TERRAN_REFINERY).nonzero()
        if not r_y.any():
            if _BUILD_REFINERY in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _VESPENE_GAS).nonzero()
                target = [unit_x[0:int(len(unit_x) / 2) - 1].mean(), unit_y[0:int(len(unit_x) / 2) - 1].mean()]
                self.refinery_built = True
                return actions.FunctionCall(_BUILD_REFINERY, [_QUEUED, target])

        if obs.last():
            reward = obs.reward

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')

            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

            self.previous_action = None
            self.previous_state = None

            self.move_number = 0

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        if obs.first():
            player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

            self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))

        factory_y, factory_x = (unit_type == _TERRAN_FACTORY).nonzero()
        factory_count = int(round(len(factory_y) / 120))  # <-- value is not accurate

        starport_y, starport_x = (unit_type == _TERRAN_STARPORT).nonzero()
        starport_count = int(round(len(starport_y) / 120))  # <-- value is not accurate


        if self.move_number == 0:
            self.move_number += 1

            current_state = np.zeros(8)
            current_state[0] = cc_count
            current_state[1] = supply_depot_count
            current_state[2] = barracks_count
            current_state[3] = obs.observation['player'][_ARMY_SUPPLY]

            hot_squares = np.zeros(4)
            enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))

                hot_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                hot_squares = hot_squares[::-1]

            for i in range(0, 4):
                current_state[i + 4] = hot_squares[i]

            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            rl_action = self.qlearn.choose_action(str(current_state))

            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT or smart_action == ACTION_BUILD_FACTORY or smart_action == ACTION_BUILD_STARPORT:
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif smart_action == _TRAIN_SCV:
                print("i want to train scvs")
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                target = [int(unit_x.mean()), int(unit_y.mean())]
                if obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and _TRAIN_SCV in \
                        obs.observation["available_actions"]:
                    print("building scvs")
                    return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])

            elif smart_action == _BUILD_REFINERY:
                if not self.refinery1_built:
                    if _BUILD_REFINERY in obs.observation["available_actions"]:
                        unit_type = obs.observation["screen"][_UNIT_TYPE]
                        unit_y, unit_x = (unit_type == _VESPENE_GAS).nonzero()
                        target = [unit_x[0:int(len(unit_x) / 2) - 1].mean(), unit_y[0:int(len(unit_x) / 2) - 1].mean()]
                        geyser1_cord = target
                        self.refinery_built = True
                        return actions.FunctionCall(_BUILD_REFINERY, [_QUEUED, target])

                elif not self.refinery2_built:
                    if _BUILD_REFINERY in obs.observation["available_actions"]:
                        unit_type = obs.observation["screen"][_UNIT_TYPE]
                        unit_y, unit_x = (unit_type == _VESPENE_GAS).nonzero()
                        target = [unit_x[int(len(unit_x) / 2) - 1:].mean(), unit_y[int(len(unit_x) / 2) - 1:].mean()]
                        geyser2_cord = target
                        self.refinery_built = True
                        return actions.FunctionCall(_BUILD_REFINERY, [_QUEUED, target])

            elif smart_action == ACTION_BUILD_MARINE:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

            elif smart_action == ACTION_BUILD_HELLION:
                if factory_y.any():
                    i = random.randint(0, len(factory_y) - 1)
                    target = [factory_x[i], factory_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

            elif smart_action == ACTION_BUILD_MEDIVAC:
                if starport_y.any():
                    i = random.randint(0, len(starport_y) - 1)
                    target = [starport_x[i], starport_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

            elif smart_action == ACTION_ATTACK:
                if _SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        elif self.move_number == 1:
            self.move_number += 1

            smart_action, x, y = self.splitAction(self.previous_action)

            # ********************************************************************************************************************

            if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if supply_depot_count < 2 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    if self.cc_y.any():
                        if supply_depot_count == 0:
                            target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
                        elif supply_depot_count == 1:
                            target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), -25)

                        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_BARRACKS:
                if barracks_count < 2 and _BUILD_BARRACKS in obs.observation['available_actions']:
                    if self.cc_y.any():
                        if barracks_count == 0:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
                        elif barracks_count == 1:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)

                        return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_FACTORY:
                if factory_count < 2 and _BUILD_FACTORY in obs.observation['available_actions']:
                    if self.cc_y.any():
                        if factory_count == 0:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
                        elif factory_count == 1:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)

                        return actions.FunctionCall(_BUILD_FACTORY, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_STARPORT:
                if starport_count < 2 and _BUILD_STARPORT in obs.observation['available_actions']:
                    if self.cc_y.any():
                        if starport_count == 0:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
                        elif starport_count == 1:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)

                        return actions.FunctionCall(_BUILD_STARPORT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            elif smart_action == ACTION_BUILD_HELLION:
                if _TRAIN_HELLION in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_HELLION, [_QUEUED])

            elif smart_action == ACTION_BUILD_MEDIVAC:
                if _TRAIN_MEDIVAC in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MEDIVAC, [_QUEUED])

            elif smart_action == ACTION_BUILD_SCV:
                if _TRAIN_SCV in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])

            # elif smart_action == ACTION_MOVE_SCV_TO_GAS_GEYSER1:
            #     unit_type = obs.observation["screen"][_UNIT_TYPE]
            #     unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
            #     target = [unit_x[0], unit_y[0]]
            #     return actions.FunctionCall(_SCV_GATHER, [_NOT_QUEUED, geyser1_cord])
            #
            # elif smart_action == ACTION_MOVE_SCV_TO_GAS_GEYSER2:
            #     unit_type = obs.observation["screen"][_UNIT_TYPE]
            #     unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
            #     target = [unit_x[0], unit_y[0]]
            #     return actions.FunctionCall(_SCV_GATHER, [_NOT_QUEUED, geyser2_cord])

            elif smart_action == ACTION_ATTACK:
                do_it = True

                if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                    do_it = False

                if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
                    do_it = False

                if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)

                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED,
                                                                  self.transformLocation(int(x) + (x_offset * 8),
                                                                                         int(y) + (y_offset * 8))])

        elif self.move_number == 2:
            self.move_number = 0

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if _HARVEST_GATHER in obs.observation['available_actions']:
                    unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()

                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [int(m_x), int(m_y)]

                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])

        return actions.FunctionCall(_NO_OP, [])