from gym.envs.registration import register

register(
    id='Dino-v0',
    entry_point='dino_gym.envs:DinoEnv',
)