import gym
from gym import spaces
import numpy as np

from selenium import webdriver
#Allows us to emulate the stroke of keyboard keys
from selenium.webdriver.common.keys import Keys
# Simulates holding down the key
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException

import json, time
import matplotlib.pyplot as plt

from PIL import Image

# Getting the canvas
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Capture screenshot
from io import BytesIO
from PIL import Image
import base64
import cv2

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 150

DINO_URL = 'chrome://dino'

def dispatchKeyEvent(driver, name, options = {}):
  options["type"] = name
  body = json.dumps({'cmd': 'Input.dispatchKeyEvent', 'params': options})
  resource = "/session/%s/chromium/send_command" % driver.session_id
  url = driver.command_executor._url + resource
  driver.command_executor._request('POST', url, body)

keys = {
  'ArrowUp': 38,
  'ArrowRight': 39,
  'ArrowDown': 40,
}

keyboard_options = [{
        "code": "ArrowRight", #event.code
        "key": "ArrowRight",    #event.key
        "nativeVirtualKeyCode": keys['ArrowRight'],
        "windowsVirtualKeyCode": keys['ArrowRight'],
    },{
        "code": "ArrowUp",
		"key": "ArrowUp",
		"nativeVirtualKeyCode": keys['ArrowUp'],
		"windowsVirtualKeyCode": keys['ArrowUp']
    },{
        "code": "ArrowDown",
        "key": "ArrowDown",
        "nativeVirtualKeyCode": keys['ArrowDown'],
        "windowsVirtualKeyCode": keys['ArrowDown']
}]


def key_down_with_options(driver, options):
    dispatchKeyEvent(driver, "rawKeyDown", options)
    dispatchKeyEvent(driver, "char", options)
    options["autoRepeat"] = True


def key_up_with_options(driver, options):
    dispatchKeyEvent(driver, "keyUp", options)



class DinoEnv(gym.Env):
    """Dino env that follows the gym interfaces."""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT):
        super(DinoEnv, self).__init__()

        chrome_options = webdriver.ChromeOptions()
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(options = chrome_options)

        # Define action and observation space
        # They must be gym.spaces objects

        # We have a discrete action space of 3.
        self.action_space = spaces.Discrete(3)

        # Buffer to store the observation.
        # Grayscale screenshot of screen_height, screen_width.
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 1),
            dtype=np.uint8,
        )

    def _screenshot(self):
        """Returns a PIL image.
        
        https://jdhao.github.io/2020/03/17/base64_opencv_pil_image_conversion/
        """
        canvas = self._driver.find_element_by_class_name("runner-canvas")
        canvas_base64 = self._driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);", canvas)

        im_bytes = base64.b64decode(canvas_base64) # binary image
        im_file = BytesIO(im_bytes) # file-like object
        img = Image.open(im_file) # PIL Image object
        return np.array(img)

    # Capture a screenshot
    def _get_observation(self):

        # Grayscale image that looks like (150, 600)
        gray = cv2.cvtColor(self._screenshot(), cv2.COLOR_BGR2GRAY)

        # Add back the rgb channel
        gray = np.expand_dims(gray, axis=2)
        return gray

    def _is_game_stopped(self):
        return not self._driver.execute_script("return Runner.instance_.playing")
    
    def _is_game_over(self):
        didDinoCrash = self._driver.execute_script("return Runner.instance_.crashed")
        return didDinoCrash
    
    def _get_score(self):
        return int(self._driver.execute_script("return Runner.instance_.distanceRan || 0"))
    
    def step(self, action):
        #Nothing
        if action == 0:
            key_up_with_options(self._driver, keyboard_options[1])
            key_up_with_options(self._driver, keyboard_options[2])
            key_down_with_options(self._driver, keyboard_options[0])
        #Jump
        elif action == 1:
            key_up_with_options(self._driver, keyboard_options[0])
            key_up_with_options(self._driver, keyboard_options[2])
            key_down_with_options(self._driver, keyboard_options[1])
        #Duck
        else:
            key_up_with_options(self._driver, keyboard_options[0])
            key_up_with_options(self._driver, keyboard_options[1])
            key_down_with_options(self._driver, keyboard_options[2])

        observation = self._get_observation()
        done = self._is_game_over()
        reward = .1 if not done else -1
        info = {'score': self._get_score()}

        time.sleep(.015)
        return observation, reward, done, info

    def _get_canvas_with_retries(self, num_retries=5):
        try:
            self._driver.get(DINO_URL)
        except:
            # Should not fail, we're loading an offline page which causes it to except.
            pass

        retries = 0
        while retries < num_retries:
            try:
                WebDriverWait(self._driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas"))
                )
                break
            except TimeoutException:
                self._driver.refresh()
                retries += 1

    def reset(self):
        self._get_canvas_with_retries()
        
        body = self._driver.find_element_by_tag_name("body")
        body.send_keys(Keys.SPACE)
        time.sleep(0.1)

        observation = self._get_observation()
        return observation
    
    def render(self, mode='human'):
        grayscale_image = self._get_observation()
        
        if mode == 'human':
            plt.imshow(grayscale_image)
            plt.axis('off')

        elif mode == 'rgb_array':
            return self._screenshot()
    
    def close(self):
        pass


#################################################################
from stable_baselines3.common.env_checker import check_env

env = DinoEnv()
# check_env(env)

#### Checking render ####
obs = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if i==500:
        env.reset()
    time.sleep(0.01)



# from stable_baselines3 import A2C
# from stable_baselines3.common.evaluation import evaluate_policy

# timesteps = 1e5

# model = A2C('CnnPolicy', env).learn(total_timesteps=int(timesteps))

# # Evaluate the agent
# # NOTE: If you use wrappers with your environment that modify rewards,
# #       this will be reflected here. To evaluate with original rewards,
# #       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# # Enjoy trained agent
# print("Play the agent")
# obs = env.reset()
# # for i in range(1000):
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

#################################################################



# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
# from stable_baselines3.common.env_util import make_vec_env

# NUM_ENVS = 4
# SAVE_FREQ = 2e5
# TIMESTEPS = 2e6

# if __name__ == "__main__":
#     frame_stack_size = 4

#     # Convert to a vectorized environment -> Stacks independent env into signle env
#     # train on n environments per step
#     venv = make_vec_env(DinoEnv, n_envs=4, vec_env_cls=SubprocVecEnv)

#     # Use previous 4 frames
#     eval_env = VecFrameStack(venv, frame_stack_size)
#     #Parallelized
#     checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path="./logs/",
#                                          name_prefix="rl_model")

#     model = PPO('CnnPolicy', eval_env)
#     model.learn(total_timesteps=TIMESTEPS, callback=[checkpoint_callback])
#     model.save('chrome_dino_model_saved')



######################################################################


    # print("Play the agent")
    # env = model.get_env()
    # obs = env.reset()

    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
        # obs, rewards, dones, info = env.step(action)
        # env.render()



# Want to compare with PPO
