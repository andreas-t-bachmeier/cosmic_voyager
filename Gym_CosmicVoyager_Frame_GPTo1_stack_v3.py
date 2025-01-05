import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import base64
import cv2
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import JavascriptException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from collections import deque  # Import deque for frame stacking

class CosmicVoyageEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(
        self,
        width=400,
        height=600,
        observation_width=100,
        observation_height=150,
        cooldown_steps=2,  # Reduced cooldown to 2
        num_stacked_frames=8
    ):
        super(CosmicVoyageEnv, self).__init__()

        # Set the desired game area dimensions
        self.game_width = width
        self.game_height = height

        # Set fixed observation dimensions
        self.observation_width = observation_width
        self.observation_height = observation_height

        # Number of frames to stack
        self.num_stacked_frames = num_stacked_frames

        # Initialize the frame buffer
        self.frames = deque(maxlen=self.num_stacked_frames)

        # Initialize the browser driver
        self._initialize_browser()

        # Define the action space (3 discrete actions: 0 - no action, 1 - move left, 2 - move right)
        self.action_space = spaces.Discrete(3)

        # Define the observation space with normalized values and frames stacked
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_stacked_frames, self.observation_height, self.observation_width),
            dtype=np.float32
        )

        # Cooldown configuration
        self.cooldown_steps = cooldown_steps  # Number of steps for cooldown
        self.cooldown_counter = 0  # Initialize cooldown counter

        # Initialize last_observation
        self.last_observation = None

        self.prev_score = 0  # Initialize previous score
        self.total_steps = 0  # Initialize total steps
        self.episode_count = 0  # Initialize episode counter

        self.last_log_time = time.time()  # Initialize last log time

    def _initialize_browser(self):
        # Initialize the Selenium WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
        options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

        # Replace with your actual ChromeDriver path
        service = Service(executable_path=r'C:\Users\Andy\Documents\GitHub\chromedriver-win64\chromedriver.exe')
        self.driver = webdriver.Chrome(service=service, options=options)

        # Adjust browser window size to accommodate the game area
        self.driver.set_window_size(self.game_width + 100, self.game_height + 200)

        # Navigate to the game URL
        self.driver.get('https://andreas-t-bachmeier.github.io/CosmicVoyager/CosmicVoyage.html')

        # Wait for the game to load completely
        time.sleep(5)

        # Set the game area size
        self._set_game_area_size(self.game_width, self.game_height)

    def _set_game_area_size(self, width, height):
        script = f"""
        var gameArea = document.getElementById('gameArea');
        if (gameArea) {{
            gameArea.style.width = '{width}px';
            gameArea.style.height = '{height}px';
            gameArea.style.maxWidth = '{width}px';
            gameArea.style.maxHeight = '{height}px';
            gameArea.style.margin = '0';  // Remove any margins
            gameArea.style.padding = '0'; // Remove any padding
        }}
        """
        self.driver.execute_script(script)
        time.sleep(1)  # Allow time for the changes to take effect

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1  # Increment episode count

        try:
            if self.episode_count % 1000 == 0:
                print("Reinitializing the browser to free up resources.")
                self._reinitialize_browser()
            else:
                # Refresh the page to reset the game
                self.driver.refresh()
            # Wait for the start button to be present
            start_button = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    start_button = WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((By.ID, 'startButton'))
                    )
                    break  # Exit the loop if successful
                except TimeoutException as e:
                    print(f"Attempt {attempt + 1} to find startButton failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(5)  # Wait before retrying
            if start_button is None:
                raise Exception("Failed to find startButton after retries.")

            # Set the game area size after refresh
            self._set_game_area_size(self.game_width, self.game_height)
            time.sleep(1)  # Allow time for size adjustment

            # Click the start button
            self.driver.execute_script("arguments[0].click();", start_button)
            time.sleep(1)  # Allow some time for the game to start
            print("Game started by clicking the start button.")

        except Exception as e:
            print("Exception occurred during reset:", e)
            self._print_console_logs()
            # Reinitialize the browser and try resetting again
            self._reinitialize_browser()
            return self.reset(seed=seed, options=options)

        # Reset cooldown counter
        self.cooldown_counter = self.cooldown_steps  # Start with cooldown expired

        # Clear the frame buffer
        self.frames.clear()

        # Capture the initial frame
        initial_frame = self._get_processed_frame()

        # Initialize the frame buffer with the initial frame
        for _ in range(self.num_stacked_frames):
            self.frames.append(initial_frame)

        # Stack frames to create the initial observation
        observation = np.stack(self.frames, axis=0)

        # Store the observation in a public attribute
        self.last_observation = observation

        self.prev_score = self._get_score()  # Initialize previous score
        self.total_steps = 0  # Reset total steps for the episode
        self.last_log_time = time.time()  # Reset last log time
        return observation, {}

    def _reinitialize_browser(self):
        # Close the existing browser
        if self.driver:
            self.driver.quit()
        # Reinitialize the browser
        self._initialize_browser()

    def step(self, action):
        self.total_steps += 1  # Increment total steps

        # Reinitialize browser every 10000 steps to free up resources
        if self.total_steps % 10000 == 0:
            print("Reinitializing the browser during step to free up resources.")
            self._reinitialize_browser()
            # Re-focus on the game
            self._focus_game_area()

        try:
            # Implement cooldown logic
            if self.cooldown_counter < self.cooldown_steps:
                # Cooldown is active; ignore the action
                action_to_take = 0  # Treat as no action
                self.cooldown_counter += 1  # Increment cooldown counter
            else:
                # Cooldown has expired; process the action
                action_to_take = action
                self.cooldown_counter = 1  # Reset cooldown counter (set to 1 to count current step)

            # Ensure the game area has focus
            self._focus_game_area()

            # Simulate key presses based on action_to_take
            if action_to_take == 1:
                self.game_area.send_keys(Keys.ARROW_LEFT)  # Simulate left arrow key press
            elif action_to_take == 2:
                self.game_area.send_keys(Keys.ARROW_RIGHT)  # Simulate right arrow key press
            elif action_to_take == 0:
                pass  # No action
        except Exception as e:
            print("Exception occurred during step:", e)
            self._print_console_logs()
            # Reinitialize the browser and try again
            self._reinitialize_browser()
            # Capture the initial frame after reinitialization
            initial_frame = self._get_processed_frame()
            for _ in range(self.num_stacked_frames):
                self.frames.append(initial_frame)
            self.last_observation = np.stack(self.frames, axis=0)
            return self.last_observation, 0, True, False, {}  # Return a termination signal

        # Sleep to control game speed
        time.sleep(0.1)  # Adjust as needed

        # Get the next frame
        frame = self._get_processed_frame()

        # Append the frame to the frame buffer
        self.frames.append(frame)

        # Stack frames to create the observation
        observation = np.stack(self.frames, axis=0)

        # Store the observation in a public attribute
        self.last_observation = observation

        # Calculate reward based on survival time
        reward = 0.1 + (self.total_steps * 0.01)  # Increased survival reward over time

        # Check if the game is over
        terminated = self._is_game_over()
        if terminated:
            reward -= 1.0  # Negative reward for losing

        # Log every 10 seconds
        current_time = time.time()
        if current_time - self.last_log_time >= 10:
            elapsed_time = current_time - self.last_log_time
            print(f"Time {elapsed_time:.2f}s: Step {self.total_steps}, Action={action}, "
                  f"Reward={reward}, Terminated={terminated}, Observation shape: {observation.shape}")
            self.last_log_time = current_time

        truncated = False  # No time limit truncation
        info = {}

        return observation, reward, terminated, truncated, info

    def _focus_game_area(self):
        # Ensure the game area has focus
        self.driver.execute_script("window.focus();")
        self.game_area = self.driver.find_element(By.TAG_NAME, 'body')
        self.driver.execute_script("arguments[0].focus();", self.game_area)

    def _get_processed_frame(self):
        # Get the screenshot of the game area element
        game_area_element = self.driver.find_element(By.ID, 'gameArea')
        frame_data = game_area_element.screenshot_as_base64
        # Decode and convert the frame to grayscale
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(frame_data), np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        # Resize the image to fixed dimensions
        img = cv2.resize(img, (self.observation_width, self.observation_height))

        # Enhance the observation using image processing techniques
        # Apply Canny edge detection
        #img = cv2.Canny(img, threshold1=50, threshold2=150)

        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Return the frame as a 2D array (H, W)
        return img

    def render(self, mode='rgb_array'):
        # Get the screenshot of the game area element
        game_area_element = self.driver.find_element(By.ID, 'gameArea')
        frame_data = game_area_element.screenshot_as_base64
        # Decode the frame to RGB
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(frame_data), np.uint8),
            cv2.IMREAD_COLOR
        )
        return img

    def _get_score(self):
        try:
            score_text = self.driver.execute_script(
                "return document.getElementById('score').innerText"
            )
            # Extract numeric value from the score text (e.g., "50 km" -> 50)
            import re
            match = re.search(r'(\d+)', score_text)
            if match:
                score = float(match.group(1))
            else:
                score = self.prev_score
        except (JavascriptException, ValueError, IndexError) as e:
            # If there's an error, use the previous score
            print(f"Error getting score: {e}")
            self._print_console_logs()
            score = self.prev_score
        return score

    def _is_game_over(self):
        try:
            game_over_visible = self.driver.execute_script("""
                return document.getElementById('gameOverScreen').classList.contains('visible');
            """)
            return game_over_visible
        except JavascriptException as e:
            # If there's an error, assume game is over
            print(f"Error checking game over state: {e}")
            self._print_console_logs()
            return True

    def _print_console_logs(self):
        # Retrieve browser logs
        logs = self.driver.get_log('browser')
        for entry in logs:
            if entry['level'] == 'SEVERE':
                print(f"Console Error: {entry['message']}")
            else:
                print(entry['message'])

    def close(self):
        # Properly close the browser and environment
        if self.driver:
            self.driver.quit()
            print("Environment closed.")

    def get_current_observation(self):
        # Return the current observation
        return self.last_observation
