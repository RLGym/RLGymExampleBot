from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

import numpy as np
from agent import Agent
from obs.advanced_obs import AdvancedObs
from rlgym_compat import GameState


class RLGymExampleBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        # FIXME Hey, botmaker. Start here:
        # Swap the obs builder if you are using a different one, RLGym's AdvancedObs is also available
        self.obs_builder = AdvancedObs()
        # Your neural network logic goes inside the Agent class, go take a look inside src/agent.py
        self.agent = Agent()
        # Adjust the tickskip if your agent was trained with a different value
        self.tick_skip = 8
        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.ticks = 0
        self.prev_time = 0
        self.observed = False
        self.acted = False
        self.expected_teammates = 0
        self.expected_opponents = 1
        self.current_obs = None
        print(f'{self.name} Ready - Index:', index)


    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.tick_multi = 120


    def reshape_state(self, gamestate, player, opponents, allies):
        """ TODO - replace me with code that handles different sized teams
        - converting to 1v1 currently """
        closest_op = min(opponents, key=lambda p: np.linalg.norm(self.game_state.ball.position - p.car_data.position))
        self.game_state.players = [player, closest_op]

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time
        ticks_elapsed = self.ticks * self.tick_multi
        self.ticks += delta

        if not self.observed:
            self.game_state.decode(packet, ticks_elapsed)
            if packet.game_info.is_kickoff_pause and not packet.game_info.is_round_active:
                ''' This would be a good time to reset the obs/action if you're using a stacking obs
                    otherwise it shouldn't really matter'''
                #self.obs_builder.reset(self.game_state)
                #self.action = np.zeros(8)
                #self.update_controls(self.action)
                pass
            player = self.game_state.players[self.index]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]
            allies = [p for p in self.game_state.players if p.team_num == self.team and p.car_id != self.index]

            if len(opponents) != self.expected_opponents or len(allies) != self.expected_teammates:
                self.reshape_state(self.game_state, player, opponents, allies)

            self.current_obs = self.obs_builder.build_obs(player, self.game_state, self.action)
            self.observed = True

        elif ticks_elapsed >= self.tick_skip-2:
            if not self.acted:
                self.action = self.agent.act(self.current_obs)
                self.update_controls(self.action)
                self.acted = True

        if ticks_elapsed >= self.tick_skip-1:
            self.ticks = 0
            self.observed = False
            self.acted = False

        return self.controls


    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
        
if __name__ == "__main__":
    print("You're doing it wrong.")

