import numpy as np
from rlgym_compat import GameState


class DiscreteAction:
    """
    Simple discrete action space. All the analog actions have 3 bins by default: -1, 0 and 1.
    """

    def __init__(self, n_bins=3):
        assert n_bins % 2 == 1, "n_bins must be an odd number"
        self._n_bins = n_bins

    def get_action_space(self):
        raise NotImplementedError("We don't implement get_action_space to remove the gym dependency")

    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        actions = actions.reshape((-1, 8)).astype(dtype=np.float32)

        # map all binned actions from {0, 1, 2 .. n_bins - 1} to {-1 .. 1}.
        actions[..., :5] = actions[..., :5] / (self._n_bins // 2) - 1

        return actions

