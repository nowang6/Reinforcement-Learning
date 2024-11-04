import os
import retro
import gym
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

def create_mario_env():
    """创建并包装马里奥环境"""
    # 创建环境
    env = retro.make(game='SuperMarioBros-Nes')
    
    # 跳帧以加速训练
    env = SkipFrame(env, skip=4)
    
    # 图像预处理
    env = ResizeObservation(env, shape=84)  # 调整大小到84x84
    env = GrayScaleObservation(env, keep_dim=True)  # 转换为灰度图
    env = FrameStack(env, num_stack=4)  # 堆叠4帧
    
    return env

def make_env():
    """创建环境的工厂函数"""
    def _init():
        env = create_mario_env()
        return env
    return _init

def train_mario():
    # 创建向量化环境
    # env = DummyVecEnv([make_env()])
    # env = VecFrameStack(env, n_stack=4)
    env_id = "SuperMarioBros-Nes"
    env = retro.make(game=env_id)
    
    # 定义模型
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./mario_tensorboard/"
    )
    
    # 设置检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./mario_checkpoints/",
        name_prefix="mario_model"
    )
    
    # 训练模型
    total_timesteps = 1000000  # 根据需要调整训练步数
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback
    )
    
    # 保存最终模型
    model.save("mario_final_model")
    
    return model

def evaluate_model(model, env, num_episodes=5):
    """评估训练好的模型"""
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()  # 显示游戏画面
            
        print(f"Episode {episode + 1} reward: {total_reward}")

if __name__ == "__main__":
    # 训练模型
    model = train_mario()
    
    # 创建评估环境
    eval_env = create_mario_env()
    
    # 评估模型
    evaluate_model(model, eval_env)