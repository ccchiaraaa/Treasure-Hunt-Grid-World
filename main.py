import pygame
import numpy as np
import random
import os

pygame.init()

GRID_SIZE = 6  # 6x6 grid
CELL_SIZE = 80  # each cell 80 pixels
INFO_HEIGHT = 120  # info panel height
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = WINDOW_SIZE + INFO_HEIGHT

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
BLUE = (0, 100, 255)
GOLD = (255, 215, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
PURPLE = (150, 0, 255)
LIGHT_BLUE = (200, 230, 255)
LIGHT_RED = (255, 200, 200)
LIGHT_GOLD = (255, 255, 200)

os.environ['SDL_VIDEO_CENTERED'] = '1'


class GridWorld:
    def __init__(self, mode="ai"):  # mode: "ai" or "human"
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_HEIGHT))
        pygame.display.set_caption("Treasure Hunt Grid World - Q-Learning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self.reset_game()
        self.mode = mode

        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.2
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        self.episode = 0
        self.total_rewards = []
        self.current_episode_reward = 0
        self.step_count = 0

    def reset_game(self):
        """Reset the game"""
        # Layout
        self.grid = [
            [' ', ' ', ' ', ' ', ' ', ' '],
            [' ', 'W', ' ', 'W', ' ', ' '],
            [' ', ' ', ' ', ' ', 'W', ' '],
            [' ', 'W', ' ', 'G', ' ', ' '],
            [' ', ' ', 'W', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' ', ' ']
        ]

        # Start position
        self.agent_pos = [0, 0]

        # Goal position
        self.goal_pos = [3, 3]  # row 3, column 3

        # Trap positions
        self.traps = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == 'W':
                    self.traps.append([i, j])

        # Game state
        self.steps = 0
        self.total_reward = 0
        self.game_over = False
        self.win = False
        self.max_steps = 50

    def get_state(self):
        """Get current state (for Q-learning)"""
        # Simplest state representation, total states = 6*6 = 36
        return (self.agent_pos[0], self.agent_pos[1])

    def get_possible_actions(self, state):
        """Get possible actions for a given state"""
        actions = []
        x, y = state

        if x > 0:  # up
            actions.append(0)
        if x < GRID_SIZE - 1:  # down
            actions.append(1)
        if y > 0:  # left
            actions.append(2)
        if y < GRID_SIZE - 1:  # right
            actions.append(3)

        return actions

    def take_action(self, action):
        """Execute action, return new state and reward"""
        new_pos = self.agent_pos.copy()

        if action == 0 and self.agent_pos[0] > 0:
            new_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < GRID_SIZE - 1:
            new_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            new_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < GRID_SIZE - 1:
            new_pos[1] += 1

        self.agent_pos = new_pos

        reward = self.calculate_reward()

        return self.get_state(), reward

    def calculate_reward(self):
        """Reward function"""
        reward = -0.1

        # Check if reached gold
        if self.agent_pos[0] == self.goal_pos[0] and self.agent_pos[1] == self.goal_pos[1]:
            reward += 10
            self.win = True
            self.game_over = True
            print("🎉 Found the gold!")

        # Check if fell into trap
        elif self.agent_pos in self.traps:
            reward -= 5
            self.game_over = True
            print("💥 Fell into a trap!")

        return reward

    def choose_action(self, state):
        """ε-greedy strategy for action selection"""
        # Get possible actions
        actions = self.get_possible_actions(state)

        # Exploration: random choice
        if random.random() < self.epsilon:
            return random.choice(actions)

        # Exploitation: choose action with highest Q-value
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in range(4)}

        q_values = {a: self.q_table[state].get(a, 0) for a in actions}
        return max(q_values, key=q_values.get)

    def update_q_table(self, state, action, reward, next_state):
        """Q-learning update formula"""
        # Initialize Q-values
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in range(4)}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in range(4)}

        # Get max Q-value for next state
        next_actions = self.get_possible_actions(next_state)
        max_next_q = max([self.q_table[next_state].get(a, 0) for a in next_actions])

        # Q-learning update formula
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def train_one_episode(self):
        """Train for one episode"""
        self.reset_game()
        state = self.get_state()
        episode_reward = 0
        step_count = 0

        while not self.game_over and self.steps < self.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return episode_reward

            # Choose action
            action = self.choose_action(state)

            next_state, reward = self.take_action(action)

            self.update_q_table(state, action, reward, next_state)

            state = next_state
            episode_reward += reward
            self.steps += 1
            step_count += 1

            # Update current episode info
            self.current_episode_reward = episode_reward
            self.step_count = step_count

            # Redraw game window after each step
            self.draw()
            pygame.time.wait(200)  # Wait 0.2 seconds to see each step
            pygame.display.update()

        # Decay exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self.episode += 1
        self.total_rewards.append(episode_reward)

        # Final draw
        self.draw()
        pygame.time.wait(500)

        return episode_reward

    def train_multiple_episodes(self, num_episodes):
        """Train for multiple episodes"""
        print(f"\n📊 Training for {num_episodes} episodes...")
        for i in range(num_episodes):
            reward = self.train_one_episode()
            print(f"  Episode {self.episode} reward: {reward:.2f}")
        print(f"✅ Training complete!")

    def draw_grid(self):
        """Draw the grid"""
        # Draw background cells
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)

                # Set color based on cell type
                if [i, j] == self.goal_pos:
                    color = GOLD
                elif [i, j] in self.traps:
                    color = RED
                else:
                    color = WHITE if (i + j) % 2 == 0 else LIGHT_BLUE

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRAY, rect, 1)

        # Draw start marker
        pygame.draw.circle(self.screen, GREEN, (CELL_SIZE // 2, CELL_SIZE // 2), 10)
        start_text = self.small_font.render("S", True, BLACK)
        self.screen.blit(start_text, (5, 5))

        # Draw agent
        if not self.game_over or self.win:
            agent_x = self.agent_pos[1] * CELL_SIZE + CELL_SIZE // 2
            agent_y = self.agent_pos[0] * CELL_SIZE + CELL_SIZE // 2

            # White outline to make blue more visible
            pygame.draw.circle(self.screen, WHITE, (agent_x, agent_y), 25)
            pygame.draw.circle(self.screen, BLACK, (agent_x, agent_y), 25, 2)

            # Blue main body
            pygame.draw.circle(self.screen, BLUE, (agent_x, agent_y), 22)
            pygame.draw.circle(self.screen, BLACK, (agent_x, agent_y), 22, 2)

            # Draw eyes
            eye_offset = 8
            pygame.draw.circle(self.screen, WHITE, (agent_x - 8, agent_y - 8), 5)
            pygame.draw.circle(self.screen, BLACK, (agent_x - 8, agent_y - 8), 2)
            pygame.draw.circle(self.screen, WHITE, (agent_x + 8, agent_y - 8), 5)
            pygame.draw.circle(self.screen, BLACK, (agent_x + 8, agent_y - 8), 2)

            # Draw smile if won
            if self.win:
                pygame.draw.arc(self.screen, BLACK,
                                (agent_x - 15, agent_y - 5, 30, 20), 0, 3.14, 3)
                for s in range(3):
                    star_x = agent_x + random.randint(-30, 30)
                    star_y = agent_y + random.randint(-30, 30)
                    pygame.draw.circle(self.screen, GOLD, (star_x, star_y), 3)

        # Draw goal marker
        goal_x = self.goal_pos[1] * CELL_SIZE + CELL_SIZE // 2
        goal_y = self.goal_pos[0] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, GOLD, (goal_x, goal_y), 15)
        goal_text = self.small_font.render("G", True, BLACK)
        goal_rect = goal_text.get_rect(center=(goal_x, goal_y))
        self.screen.blit(goal_text, goal_rect)

    def draw_info(self):
        """Draw information panel"""
        info_y = WINDOW_SIZE + 10

        # Background
        pygame.draw.rect(self.screen, DARK_GRAY,
                         (0, WINDOW_SIZE, WINDOW_SIZE, INFO_HEIGHT))

        # Info text
        info_texts = [
            f"Episode: {self.episode}",
            f"Steps: {self.steps}/{self.max_steps}",
            f"Epsilon: {self.epsilon:.3f}",
            f"Q-table size: {len(self.q_table)}",
            f"Current reward: {self.current_episode_reward:.2f}"
        ]

        if self.total_rewards:
            avg_reward = sum(self.total_rewards[-10:]) / min(10, len(self.total_rewards))
            info_texts.append(f"Avg reward (last 10): {avg_reward:.2f}")

        for i, text in enumerate(info_texts):
            text_surface = self.small_font.render(text, True, WHITE)
            self.screen.blit(text_surface, (10, info_y + i * 20))

        status_y = info_y
        status_x = WINDOW_SIZE - 250

        if self.game_over:
            if self.win:
                status = "VICTORY!"
                color = GOLD
            else:
                status = "GAME OVER"
                color = RED
        else:
            status = "▶ RUNNING"
            color = GREEN

        status_surface = self.font.render(status, True, color)
        self.screen.blit(status_surface, (status_x, status_y))

        hint = "R:Reset | T:Train 1 | A:Train 10 | Q:Quit"
        hint_surface = self.small_font.render(hint, True, WHITE)
        self.screen.blit(hint_surface, (status_x, status_y + 40))

        mode_text = f"Mode: {'Human' if self.mode == 'human' else 'AI Training'}"
        mode_surface = self.small_font.render(mode_text, True, LIGHT_GOLD)
        self.screen.blit(mode_surface, (status_x, status_y + 60))

    def draw(self):
        """Draw entire interface"""
        self.screen.fill(WHITE)
        self.draw_grid()
        self.draw_info()
        pygame.display.flip()
        pygame.display.update()  # Force update

    def handle_human_input(self):
        """Handle keyboard input for human mode"""
        keys = pygame.key.get_pressed()
        action = None

        if keys[pygame.K_UP]:
            action = 0
        elif keys[pygame.K_DOWN]:
            action = 1
        elif keys[pygame.K_LEFT]:
            action = 2
        elif keys[pygame.K_RIGHT]:
            action = 3

        if action is not None and not self.game_over:
            next_state, reward = self.take_action(action)
            self.total_reward += reward
            self.steps += 1
            self.current_episode_reward = self.total_reward

            action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
            print(f"Action: {action_names[action]}, Reward: {reward:.2f}")
            self.draw()

    def run(self):
        """Main game loop"""
        running = True
        training_in_progress = False

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset
                        self.reset_game()
                        self.current_episode_reward = 0
                        training_in_progress = False
                        print("\n🔄 Game reset")
                        self.draw()

                    elif event.key == pygame.K_t:  # Train 1 episode
                        if self.mode == "ai":
                            training_in_progress = True
                            self.train_one_episode()
                            training_in_progress = False
                        else:
                            print("Human mode: Use arrow keys to move")

                    elif event.key == pygame.K_a:  # Train 10 episodes
                        if self.mode == "ai":
                            training_in_progress = True
                            self.train_multiple_episodes(10)
                            training_in_progress = False
                        else:
                            print("Human mode: Use arrow keys to move")

                    elif event.key == pygame.K_q:  # Quit
                        running = False

            if self.mode == "human" and not training_in_progress:
                self.handle_human_input()

            self.draw()
            self.clock.tick(10)  # 10 FPS

        pygame.quit()

        if self.total_rewards:
            print("\n Training Statistics:")
            print(f"Total episodes: {len(self.total_rewards)}")
            print(f"Average reward: {np.mean(self.total_rewards):.2f}")
            print(f"Max reward: {np.max(self.total_rewards):.2f}")
            print(f"Final epsilon: {self.epsilon:.3f}")
            print(f"Q-table states: {len(self.q_table)}")


if __name__ == "__main__":
    print("=" * 60)
    print(" TREASURE HUNT GRID WORLD - Q-Learning Game")
    print("=" * 60)
    print("\nGame Rules:")
    print("  🔵 Blue dot = Your agent")
    print("  🟨 Gold cell = Goal (treasure)")
    print("  🟥 Red cell = Trap")
    print("  🟩 Green dot = Start position")
    print("\nControls:")
    print("  ⬆⬇⬅➡ Arrow keys = Manual control (Human mode)")
    print("  R = Reset game")
    print("  T = Train 1 episode (AI mode)")
    print("  A = Train 10 episodes (AI mode)")
    print("  Q = Quit")
    print("\nSelect mode:")
    print("  1 - Human control (play with arrow keys)")
    print("  2 - AI training mode (watch Q-learning learn)")

    choice = input("\nEnter 1 or 2: ")

    if choice == "1":
        game = GridWorld(mode="human")
        print("\n Human mode activated! Use arrow keys to move the blue dot.")
        print("Find the gold (G) and avoid red traps!")
    else:
        game = GridWorld(mode="ai")
        print("\n AI training mode activated!")
        print("Press T to train for 1 episode, or A to train for 10 episodes.")

    game.run()