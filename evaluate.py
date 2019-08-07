from sensor import SensorEnv
import pandas as pd
import numpy as np
import random
import sys
import os
def build_env(env_single, nenv=None):
    import multiprocessing
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.bench import Monitor
    from baselines import logger
    from baselines.common import set_global_seeds
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = nenv or ncpu
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = random.seed(0)
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env_single.seed(seed + 10000*mpi_rank + rank if seed is not None else None)
            env = Monitor(env_single,
                          logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(rank)),
                          allow_early_resets=True)
            return env
        return _thunk
    set_global_seeds(seed)
    if nenv > 1: return SubprocVecEnv([make_env(i+0) for i in range(nenv)])
    else: return DummyVecEnv([make_env(0)])

def train(num_timesteps, lr, num_layers, num_hidden,env_single, nenv, nsteps, lam, gamma):
    import baselines.ppo2.ppo2 as alg_module
    from baselines import logger
    logger.set_level(logger.DISABLED)
    env = build_env(env_single, nenv)
    alg_kwargs={'network':'mlp', 'num_layers':num_layers, 'num_hidden':num_hidden}
    model = alg_module.learn(env= env,
                  total_timesteps=num_timesteps,
                  nsteps=nsteps,
                  nminibatches=4,
                  lam=lam, gamma=gamma, noptepochs=4, log_interval=1,
                  ent_coef=.01,
                  lr=lambda f : f * lr,
                  cliprange=lambda f : f * 0.1,
                  **alg_kwargs
                 )
    return model

def evaluate(params, model_path):
    import tensorflow as tf
    from baselines.common.tf_util import get_session
    tf.reset_default_graph()
    sess= get_session()
    sess.close()
    history=[]
    env =SensorEnv(cont_actions=params['cont_actions'],
                   forecast_days=params['forecast_days'],
                   sparse=params['sparse'],
                   episode_len=params['episode_len'],
                   damping_factor=params['damping_factor'],
                   init_buffer_state=params['init_buffer_state'],
                   failure_penalty=params['failure_penalty'],
                   solar_file='data/Tokyo_2011-eng.txt'
                  )
    model = train(num_timesteps=0,
                  lr=params['lr'], 
                  num_layers=params['num_layers'], 
                  num_hidden=params['num_hidden'],
                  env_single=env,
                  nenv=1,
                  nsteps=params['nsteps'],
                  lam=params['lam'],
                  gamma=params['gamma'])
    model.load(model_path)
    ob= env.reset(episode_start= env.daterange[0], buffer_state=10)
    for _ in range(364*24):
        action, _, _, _ = model.step(ob)
        next_ob, r, done, info = env.step(action)
        history.append(info)
        ob = next_ob
    history = pd.DataFrame(history)
    history.index= history['timestamp']
    history['std']= history['duty_cycle'].rolling(window=24, min_periods=1).std()
    history['std'].fillna(0, inplace=True)
    variance = history['std'].mean()
    wasted_energy = abs(history['energy_wasted'].sum()/env.B_max)
    failures = history['failure'].sum()
    utilized_energy= (history['duty_cycle'].sum()*5)/env.B_max
    return [history, variance, wasted_energy, failures, utilized_energy]