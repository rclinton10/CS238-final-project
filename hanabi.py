from hanabi_learning_environment import rl_env
import json
from hanabi_learning_environment.agents.random_agent import RandomAgent
from hanabi_learning_environment.agents.simple_agent import SimpleAgent

# Map agent names to their respective classes
AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent}

# Extracts and encodes the relevant parts of a player's observation into a dictionary format.
# TODO see which of these we actually need; remove the others
def encode_state(observation):
    state = {
        "visible_hands": observation["observed_hands"],  # Other players' hands
        "fireworks": observation["fireworks"],          # Current state of played stacks
        "discard_pile": observation["discard_pile"],    # Discarded cards
        "hint_tokens": observation["information_tokens"],  # Remaining hint tokens
        "life_tokens": observation["life_tokens"],      # Remaining life tokens
    }
    return state

class Runner(object):
  """Runner class."""

  def __init__(self):
    """Initialize runner."""
    self.agent_config = {'players': 2}
    self.environment = env = rl_env.HanabiEnv(config={
            "colors": 5,
            "ranks": 5,
            "players": 2,
            "hand_size": 5,
            "max_information_tokens": 8,
            "max_life_tokens": 3,
        })
    self.agent_class = AGENT_CLASSES['SimpleAgent'] # default to simple agent

  def run(self):
    """Run episodes."""
    simulations = []
    for episode in range(1000): # TODO Increase/decrease depending on how many episodes we want to do 
        """
        Before we proceed with each episode, we want to reset the environment for a new game.
        The reset function returns a dictionary containing the initial state of the game:
            - Observed hands (other players' cards visible to the current player)
            - Fireworks stacks
            - Discard pile
            - Information and life tokens
            - Legal moves for the players
        """
        observations = self.environment.reset()
        agents = [self.agent_class(self.agent_config) for _ in range(2)] # Initialize agents for the game (one agent per player)
        done = False
        while not done:
            for agent_id, agent in enumerate(agents):
                observation = observations['player_observations'][agent_id]
                action = agent.act(observation)

                # Validate the action: Only the current player can take actions
                if observation['current_player'] == agent_id:
                    assert action is not None
                    current_player_action = action
                else:
                    assert action is None

            observations, reward, done, unused_info = self.environment.step(current_player_action)
            simulation = {
                "state": encode_state(observation),
                "action": current_player_action,
                # "agent": observation['current_player'], # Not sure if this is needed 
                "reward": reward,
                "next_state": encode_state(observations["player_observations"][0]),
                # "done": done # Also not sure if this is needed
            }
            simulations.append(simulation)
    with open("hanabi_simulations.json", "w") as file:
        json.dump(simulations, file, indent=2)
    print("Just dumped all of the game simulations to hanabi_simulations.json!")

if __name__ == "__main__":
  runner = Runner()
  runner.run()
