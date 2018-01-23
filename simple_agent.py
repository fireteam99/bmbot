from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time
import sys

# Functions
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_BARRACKS = 21
_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLYDEPOT = 19
_TERRAN_SCV = 45
_VESPENE_GEYSER = 342

# Parameters
_PLAYER_SELF = 1
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_NOT_QUEUED = [0]
_QUEUED = [1]


class SimpleAgent(base_agent.BaseAgent):
    base_top_left = None
    supply_depot_built = False
    scv_selected = False
    barracks_built = False
    barracks_selected = False
    barracks_rallied = False
    army_selected = False
    army_rallied = False
    refinery_built = False

    def closestVespeneGeyser(self, base_cord_x, base_cord_y, geysers_x, geysers_y):
        smallest = sys.maxsize
        closest_geyser_index = 0
        list_len = len(geysers_x)
        for i in range(0,list_len):
            geyser_x_cord = geysers_x[i]
            geyser_y_cord = geysers_y[i]
            distance = ((base_cord_x - geyser_x_cord)**2 + (base_cord_y - geyser_y_cord)**2)
            if(distance < smallest):
                smallest = distance
                closest_geyser_index = i
        return [geysers_x[closest_geyser_index], geysers_y[closest_geyser_index]]


    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        time.sleep(0.1)

        if self.base_top_left is None:
            player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31

        if not self.supply_depot_built:
            if not self.scv_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                target = [unit_x[0], unit_y[0]]

                self.scv_selected = True

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                print("cmd center is at: x=" + str(unit_x.mean()) + " y=" + str(unit_y.mean()))

                target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)

                self.supply_depot_built = True

                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])
        elif not self.refinery_built:
            if not self.scv_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                target = [unit_x[0], unit_y[0]]

                self.scv_selected = True

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif _BUILD_REFINERY in obs.observation["available_actions"]:
                print("attempting to build refinery")
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _VESPENE_GEYSER).nonzero()
                print(str(unit_x))
                print(str(unit_y))
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                command_center_y, command_center_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                target = unit_x[23], unit_y[23]
                #target = self.closestVespeneGeyser(command_center_x[0], command_center_y[0], unit_x, unit_y)

                self.refinery_built = True

                return actions.FunctionCall(_BUILD_REFINERY, [_NOT_QUEUED, target])



        elif not self.barracks_built and self.refinery_built:
            if _BUILD_BARRACKS in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)

                self.barracks_built = True

                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
        elif not self.barracks_rallied:
            if not self.barracks_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]

                    self.barracks_selected = True

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            else:
                self.barracks_rallied = True

                if self.base_top_left:
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])

                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 46]])
        elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and _TRAIN_MARINE in \
                obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        elif not self.army_rallied:
            if not self.army_selected:
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.army_selected = True
                    self.barracks_selected = False

                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                self.army_rallied = True
                self.army_selected = False

                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])

                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])

        return actions.FunctionCall(_NOOP, [])