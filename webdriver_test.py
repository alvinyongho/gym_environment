# Tests whether we installed dependencies properly

import gym
from gym import spaces
import numpy as np

from selenium import webdriver
#Allows us to emulate the stroke of keyboard keys
from selenium.webdriver.common.keys import Keys
# Simulates holding down the key
from selenium.webdriver.common.action_chains import ActionChains
import time


# Getting the canvas
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Capture screenshot
from io import BytesIO
from PIL import Image
import base64
import cv2

# Manage the state
from collections import deque

# Debug
from matplotlib.pyplot import imshow
DINO_URL = 'https://chromedino.com/'

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--mute-audio")
driver = webdriver.Chrome(options = chrome_options)
driver.get(DINO_URL)
