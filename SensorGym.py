import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
import astral
class SensorGymEnv(gym.Env):
    """
        SensorGym is an interface to reinforcement learning tasks in 
        energy-harvesting IoT. It is based on a realistic IoT node specification
        (scaled up version of TMote Sky node).
        
        **INPUTS:** to Initialize Env
        solar_file: solar radiation data 
        city: to calculate zenith & azimuth
        forecast_days: days in include in weather forecast
        w_forcast_err: prediction error in weather forecast
        episode_len: maximum episode length in days
        time_unit: time unit of one epoch
        B_max: maximum capacity of the energy buffer (mWh)
        B_fail: minimun energy in the buffer at which the IoT node shutdown  (mWh)
        init_buffer_state: initial percentage of energy at which to start training
        min_duty_cycle: minimal operation duty cycle of an IoT node
        damping_factor: factor int the reward to reduce variance 
        sparsity: after how many time units to give the reward
        failure_penalty: penality if the energy buffer reached B_fail
        cont_actions: True if you want the env to accept continuous actions, otherwise discrete

        **OBSERVATION SPACE**
        The observation space consists of  the  energy buffer,
        harvested energy, epected weather, and the zenith at a
        specific time:
        [self.buffer_state, e_harvest, w_forecast, zenith]
        

        **ACTIONS:**
        The action correspond to setting the operation duty cycle of an IoT node 
        (between min_ & max_duty_cycle)
    """
    def __init__(self,solar_file='solar_data/Tokyo_2010.csv', 
                 city='Tokyo', forecast_days=3,w_forcast_err=0.2,
                 episode_len=365,time_unit='h', delta=1,
                 B_max=40000, B_fail=0,init_buffer_state=60, min_duty_cycle=20.0,
                 damping_factor=0.01, sparsity=24, failure_penalty=-100,cont_actions=True ):
        
        self.damping_factor = damping_factor
        self.sparse = sparsity
        self.sparse_counter = 1
        self.B_max = B_max
        self.delt = delta
        self.time_unit=time_unit
        self.time_delta = pd.Timedelta(str(delta)+time_unit)
        self.episode_len = episode_len*24
        self.init_buffer_state = init_buffer_state
        self.last_duty_cycles = []
        self.cont_actions = cont_actions
        self.failure_penalty = failure_penalty*sparsity
        self.B_fail= B_fail
        self.min_duty_cycle = min_duty_cycle
        self.max_duty_cycle = 100.0
    
        #creating env's external context(solar and weather data, time)
        self.solar_context= self._create_solar_context(forecast_days,solar_file, w_forcast_err, city)
        self.daterange =self.solar_context['w_forecast'].index
        
        # Setting action_space
        if self.cont_actions:
            self.action_space = spaces.Box(low=self.min_duty_cycle, high=self.max_duty_cycle , shape=(1,),dtype= np.float32)
        else:
            self.action_space = spaces.Discrete(11)
        
        # Setting observation_space
        max_e_buffer = 100.0
        max_e_harvest = self.solar_context['dataframe']['e_harvest'].max()
        max_w_forecast =self.solar_context['w_forecast'].max()
        max_zenith = self.solar_context['dataframe']['zenith'].max()
        self.high_state = np.array([max_e_buffer,max_e_harvest, max_w_forecast, max_zenith])
        self.low_state = np.array([0.0, 0.0, 0.0, -max_zenith])
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype= np.float32)
    
        self.seed()
        self.reset()
    

    
    def step(self, action):
        """Run one timestep of the environment's dynamics. 
        Args:
            action (object): duty cycle provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended or the energy buffer reached B_fail 
            info (dict): contains auxiliary diagnostic information 
        """
        #make sure the action is legal(within the action space)
        assert not np.isnan(action)
        action = np.squeeze(action)
        if self.cont_actions:
            duty_cycle = np.clip(action, self.min_duty_cycle, self.max_duty_cycle)
        else:
            assert self.action_space.contains(action), "%r (%s) invalied"% (action, type(action))
            duty_cycle = (action)
        
        #get external environment's context at the current timestep (self.t)
        e_harvest, w_forecast, zenith = self._solar_intake(self.t, self.solar_context)
        
        # calculate the consumed energy
        e_consumed = duty_cycle*5# based on TMote Sky node spec (mWh)
        buffer_state_next, energy_wasted, failure = self._energy_buffer(self.t, e_harvest, e_consumed, self.buffer_state)
        self.buffer_state = buffer_state_next
        
         # calculate the reward based ont the reward function
        self.last_duty_cycles.append(duty_cycle)
        if self.sparse == 1:
            reward = int(self.last_duty_cycles[-1] - self.damping_factor*sum([(t-s)**2 for s , t in zip(self.last_duty_cycles, self.last_duty_cycles[1:])]))
            del self.last_duty_cycles[:-1]
        elif (self.sparse_counter%self.sparse) == 0:
            reward = int(sum(self.last_duty_cycles) - self.damping_factor*sum([(t-s)**2 for s , t in zip(self.last_duty_cycles, self.last_duty_cycles[1:])]))
            self.sparse_counter = 1
            self.last_duty_cycles = [] 
        else:
            reward = 0
            self.sparse_counter +=1
            
        #if the energy buffer reached B_fail, give penalty and end the episode.    
        if failure:
            duty_cycle = 0
            reward = self.failure_penalty
            done = True
            
        #Increment the timestep of the environment's dynamics
        if (self.t.is_year_end):
            self.t = self.daterange[0]
        else:
            self.t += self.time_delta
        
        # check whether the episode has ended, warns the agent
        self.remaining_epochs -=1 
        done = self.remaining_epochs <=0 
        if done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment \
                             has already returned done = True. ")
                self.steps_beyond_done +=1
        
        # 
        self.ob = np.array([self.buffer_state, e_harvest, w_forecast, zenith])
        info = {'timestamp': self.t-self.time_delta, 'buffer': self.buffer_state, 'e_harvest': e_harvest, 
                'w_forecast': w_forecast,'reward': reward, 'consumption': e_consumed, 'duty_cycle': duty_cycle,'action':action, 
                 'energy_wasted': energy_wasted,'failure': failure, 'zenith': zenith
               }
        return (self.ob, reward, done, info)

    
    def reset(self, episode_start=None, buffer_state=None):
        """Resets the state of the environment and returns an initial observation.
        Returns: 
            observation (object): the initial observation.
        """
        #check where to in the year to start the episode, otherwise make it random.
        if episode_start is None:
            index = np.random.randint(len(self.daterange))
            self.t = self.daterange[index]
        else:
            self.t = episode_start
         
        if buffer_state is None:
            self.buffer_state = self.init_buffer_state# np.random.randint(5,100)
        else:
            self.buffer_state = buffer_state
            
        e_harvest, w_forecast, zenith = self._solar_intake(self.t, self.solar_context)
        self.ob = np.array([self.buffer_state, e_harvest,  w_forecast, zenith])
        
        self.remaining_epochs = self.episode_len
        self.steps_beyond_done = None
        
        return self.ob       
    
    def _create_solar_context(self,forecast_days, solar_file, w_forcast_err, city):
        solar_panel_config = {'voltage_t': 7.32, 'area_j': 0.0165, 'efficiency_j': 0.15}
        dfx = pd.read_csv(solar_file, index_col=0,parse_dates=True)
        a = astral.Astral()
        location = a[city]
        lat = location.latitude
        lon = location.longitude
        dfx['zenith'] =dfx.index.to_series().apply(lambda timestamp: a.solar_zenith(timestamp, lat, lon))
        dfx['zenith']= (dfx['zenith'] - dfx['zenith'].mean())/dfx['zenith'].std()
        dfx['azimuth'] =dfx.index.to_series().apply(lambda timestamp: a.solar_azimuth(timestamp, lat, lon))
        dfx['azimuth']= (dfx['azimuth'] - dfx['azimuth'].mean())/dfx['azimuth'].std()
        dfx.loc[dfx.index[0]-pd.to_timedelta(1, unit='h')] = dfx.loc[dfx.index[-1]]
        dfx = dfx.sort_index()[:-1]
        area = solar_panel_config['area_j']
        efficiency = solar_panel_config['efficiency_j']
        dfx['e_harvest'] = dfx['a']*area*1E9*efficiency/(60*60)
        dfx['w_forecast'] = np.random.uniform(low=(dfx['a']-(dfx['a']*w_forcast_err)), high=(dfx['a']+(dfx['a']*w_forcast_err)))
        dfx['w_forecast'] = dfx['w_forecast']*area*1E9*efficiency/(60*60)
        dfx.drop('a', axis=1, inplace=True)
        forecast_day= dfx['w_forecast'].resample('D').sum()
        daterange = forecast_day.index
        for i in range(forecast_days):
            forecast_day.loc[forecast_day.index[-1]+1]= forecast_day.loc[forecast_day.index[i]]
        forecast= forecast_day.rolling(str(forecast_days)+'D').sum().shift(-(forecast_days-1))[:-forecast_days]
        return {'dataframe': dfx, 'w_forecast': forecast}
        
    
    def _solar_intake(self, t, solar_context):
        dfx = solar_context['dataframe']
        index = dfx.index.get_loc(t, method = 'nearest')
        zenith = dfx['zenith'].iloc[index]
        e_harvest = dfx['e_harvest'].iloc[index]
        w_forecast = solar_context['w_forecast']
        w_forecast = w_forecast[t.date()]
        return e_harvest, w_forecast, zenith 
    
    def _energy_buffer(self, t, e_harvest, e_consumed, buffer_state):
        f = 100/self.B_max
        buffer_state_next = buffer_state +f*(e_harvest - e_consumed)
        energy_wasted = 0
        failure = False
        if buffer_state_next > 100:
            energy_wasted = -(buffer_state_next -100)/f
            buffer_state_next = 100
        elif buffer_state_next < self.B_fail*f:
            failure = True
            buffer_state_next = buffer_state
        return buffer_state_next, energy_wasted, failure
    
    def render(self, mode='human'):
        """Renders the environment.
        Not implemented
        """
        if mode =='rgb_array':
            print('return np.array(...), RGB fram suitable for video')
        elif mode == 'human':
            print('pop up window and render')
        else:
            super(SensorEnv, self).render(mode=mode) #just raise an exception
            
    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def close(self):
        """
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        print('Perform any necessary cleanup')
        