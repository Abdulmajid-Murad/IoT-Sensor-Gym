from SensorGym import SensorGymEnv
import pandas as pd
import numpy as np
import random
import sys
import os
from timeit import default_timer as timer
from tqdm import tqdm

def build_env(env_single, nenv=None):
    import multiprocessing
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.bench import Monitor
    from baselines import logger
    from baselines.common import set_global_seeds

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
                          filename=None,#logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(rank)),
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
                  lam=lam, gamma=gamma, noptepochs=4, log_interval=1000,
                  ent_coef=.01,
                  lr=lambda f : f * lr,
                  cliprange=lambda f : f * 0.1,
                  **alg_kwargs
                 )
    return model

def evaluate(model, env):
    history=[]
    ob= env.reset(episode_start= env.daterange[0], buffer_state=60)
    for _ in range(env.episode_len-24):
        action, _, _, _ = model.step(ob)
        next_ob, r, done, info = env.step(action)
        history.append(info)
        ob = next_ob
    
    history = pd.DataFrame(history)
    history['std']= history['duty_cycle'].rolling(window=24, min_periods=1).std()
    history['std'].fillna(0, inplace=True)
    variance = history['std'].mean()
    wasted_energy = abs(history['energy_wasted'].sum()/env.B_max)
    failures = history['failure'].sum()
    total_reward = history['reward'].sum()
    utilized_energy= (history['duty_cycle'].sum()*5)/env.B_max
    return [variance,wasted_energy, failures, total_reward, utilized_energy]
    
def random_objective(params, iteration):
    import tensorflow as tf
    from baselines.common.tf_util import get_session
    start = timer()
    env =SensorGymEnv(cont_actions=params['cont_actions'],
                   forecast_days=params['forecast_days'],
                   sparsity=params['sparse'],
                   episode_len=params['episode_len'],
                   damping_factor=params['damping_factor'],
                   init_buffer_state=params['init_buffer_state'],
                   failure_penalty=params['failure_penalty']
                  )
    model = train(num_timesteps=params['num_timesteps'],
                  lr=params['lr'], 
                  num_layers=params['num_layers'], 
                  num_hidden=params['num_hidden'],
                  env_single=env,
                  nenv=params['nenv'],
                  nsteps=params['nsteps'],
                  lam=params['lam'],
                  gamma=params['gamma'])
    
    save_path = os.path.join('saved_models2', "{}".format(iteration))
    model.save(save_path)
    tf.reset_default_graph()
    sess= get_session()
    sess.close()
    
    eval_model = train(num_timesteps=0,
                  lr=params['lr'], 
                  num_layers=params['num_layers'], 
                  num_hidden=params['num_hidden'],
                  env_single=env,
                  nenv=1,
                  nsteps=params['nsteps'],
                  lam=params['lam'],
                  gamma=params['gamma'])
    eval_model.load(save_path)
    eval_env = SensorGymEnv(cont_actions=params['cont_actions'],
                   forecast_days=params['forecast_days'],
                   sparsity=params['sparse'],
                   episode_len=365,
                   damping_factor=params['damping_factor'],
                   init_buffer_state=60,
                   failure_penalty=params['failure_penalty'],
                   solar_file='solar_data/Tokyo_2011.csv'
                  )
    variance,wasted_energy, failures, total_reward, utilized_energy = evaluate(eval_model, eval_env)
    tf.reset_default_graph()
    sess= get_session()
    sess.close()
    end = timer()
    return [variance,wasted_energy, failures, total_reward, utilized_energy, params, iteration, end-start]

def main():
    MAX_EVALS = 100
    param_grid = {
        'cont_actions': [True,False],
        'forecast_days': [1,2,3],
        'sparse': [1,24],
        'episode_len':[364],
        'damping_factor':list(np.logspace(-1, -3, num=100)),#[0.01],
        'init_buffer_state':[60],
        'failure_penalty':[-500],
        'lr': list(np.logspace(-1, -3, num=100)),
        'num_layers': [2],
        'num_hidden': [32],
        'nsteps':[256],
        'gamma':[0.99],
        'lam':[0.95],
        'nenv':[40],
        'num_timesteps':[int(1e7)]
    }

    random_results = pd.DataFrame(columns=['variance','wasted_energy','failures',
                                           'total_reward','utilized_energy',
                                           'params','iteration','time'], 
                             index=list(range(MAX_EVALS)))
    for i in tqdm(range(MAX_EVALS)):
        params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
        results_list = random_objective(params, i)
        random_results.loc[i, :] = results_list
        random_results.to_csv('results.csv')    
   
if __name__ == "__main__":
    main()