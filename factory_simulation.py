import simpy
import random

class Factory_Simulation(): 
    """ A simple manufacturing simulation for scheduling with five stations and two products, 
    created using SimPy with an n OpenAI gym-like interface for Reinforcement Learning. """

    def __init__(self, simulation_duration=120, time_step=1, setup_time=0, reward_value=0):
        """ Constructor method for Factory_Simulation class. """

        # Set up duration for each simulation run (returns terminal state afterwards) and time steps 
        self.simulation_duration = simulation_duration
        self.time_step = time_step
        self.next_time_stop = 0 # used to iterate over env

        # Set up fixed factory parameters
        self.setup_time = setup_time
        self.buffer_capacity = 999999 # virtually unlimited
        self.initial_buffer = 0 # factory starts empty
        self.initial_reward = reward_value
        
        # Set up manufacturing times for all five machines for product A and B 
        self.process_time_a = [5, 4, 2, 2, 1] # m1, m2, m3, m4, m5
        self.process_time_b = [1, 2, 3, 4, 5] # ...

        # Set up standard deviation for process time variability
        self.default_std_dev = .0

        # Set up list to store fifo rule for one simulation run
        self.fifo_list = []
        # Set up list to save all actions performed by the agent
        self.all_actions = []

        # Set up time to get raw or semi materials
        self.time_to_get_material = 1
        # Set up costs as negative rewards for machine runs (cost per t)
        self.cost_for_machining = 0
        # Set up costs as negative rewards for storing material 
        self.cost_for_storage = 1
        # Set up price for raw material 
        self.cost_for_raw_material = 10

        # Set up counter for finished jobs
        self.finished_jobs_m1 = 0
        self.finished_jobs_m2 = 0
        self.finished_jobs_m3 = 0
        self.finished_jobs_m4 = 0
        self.finished_jobs_m5 = 0

        # Set up counter for reward according to product mix
        self.consecutive_production_of_a = 0 
        self.consecutive_production_of_b = 0
        self.consecutive_idle_states = 0

        # Set up A, R, S for rl-agent
        self.actions = [0, 1, 2] # [ produce A, produce B, wait ]
        self.reward = self.initial_reward # using neg. value to punish idle states
        self.state = dict()

        # Set up observation and action space sizes
        self.observation_size = 12 # one for each container
        self.action_size = len(self.actions)

    def _get_next_job(self, finished_jobs):
        """ Returns the next job for a machine according to the fifo list. """
        return self.fifo_list[finished_jobs]

    def _apply_variability(self, process_time):
        """ Returns the process time under the influence of variability (gaussian normal distribution). """
        # Apply
        return random.gauss(process_time, process_time*self.default_std_dev)

    def _calculate_sales_value(self, product_type):
        """ Returns the current sales value of the finished products A and B. """
        
        # Value higher amounts of finished goods (regardless of type)
        # return 100 + self.env.material_m5_a.level + self.env.material_m5_b.level

        # Value mix of products in terms of the over-all reward
        if product_type=='A':
            self.consecutive_production_of_a += 1
            self.consecutive_production_of_b = 0
            #return 110 - (10*self.consecutive_production_of_a)
            return 100 * (0.9 ** (self.consecutive_production_of_a-1))
        elif product_type=='B':
            self.consecutive_production_of_b += 1
            self.consecutive_production_of_a = 0
            #return 110 - (10*self.consecutive_production_of_b)
            return 100 * (0.9 ** (self.consecutive_production_of_b-1))

        else:
            raise ValueError('{} --> requested type is not in the list of allowed product types [A, B]'.format(product_type)) 
        
        # Simply return a default value for finished goods
        # return 100 

    def _run_machine_1(self):
        """ Runs the manufacutring programm of Machine 1 and needs to be called as a SimPy Process. """
        while True: 
            
            # Check if an order has to be executed
            jobs = len(self.fifo_list)
            # jobs == 0: no action yet, or only action 2
            # jobs == finished jobs: no more jobs available 
            if (jobs==0) or (jobs==self.finished_jobs_m1):
                yield self.env.timeout(self.time_step)
            else:
                # Check which order has to be executed
                next_job = self._get_next_job(self.finished_jobs_m1)

                # Check if a machine set-up is necessary due to product changes (A to B, or B to A)
                if jobs==1: # initial set-up required
                    yield self.env.timeout(self.setup_time)
                elif self.fifo_list[self.finished_jobs_m1-1] != next_job: # changing product
                    yield self.env.timeout(self.setup_time)

                if next_job=='A': 
                    # simulate 'getting' the material 
                    yield self.env.timeout(self.time_to_get_material)
                    # take the material for A from the prior buffer
                    yield self.env.raw_material_a.get(1) 
                    # processing the job
                    self.reward -= self.cost_for_machining * self.process_time_a[0]
                    yield self.env.timeout(self._apply_variability(self.process_time_a[0])) 
                    # place the finished good in next buffer
                    yield self.env.material_m1_a.put(1) 
                    #increase job counter by 1
                    self.finished_jobs_m1 += 1

                elif next_job=='B':
                    # simulate 'getting' the material 
                    yield self.env.timeout(self.time_to_get_material)
                    # take the material for B from the prior buffer
                    yield self.env.raw_material_b.get(1) 
                    # processing the job
                    self.reward -= self.cost_for_machining * self.process_time_b[0]
                    yield self.env.timeout(self._apply_variability(self.process_time_b[0])) 
                    # place the finished good in next buffer
                    yield self.env.material_m1_b.put(1) 
                    #increase job counter by 1
                    self.finished_jobs_m1 += 1

                else: 
                    raise ValueError('{} --> requested job is not in the list of allowed jobs [A, B]'.format(next_job))         

    def _run_machine_2(self):
        """ Runs the manufacutring programm of Machine 2 and needs to be called as a SimPy Process. """
        while True: 
            
            # Check if an order has to be executed
            jobs = len(self.fifo_list)
            # jobs == 0: no action yet, or only action 2
            # jobs == finished jobs: no more jobs available 
            if (jobs==0) or (jobs==self.finished_jobs_m2):
                yield self.env.timeout(self.time_step)
            else:
                # Check which order has to be executed
                next_job = self._get_next_job(self.finished_jobs_m2)

                # Check if a machine set-up is necessary due to product changes (A to B, or B to A)
                if jobs==1: # initial set-up required
                    yield self.env.timeout(self.setup_time)
                elif self.fifo_list[self.finished_jobs_m2-1] != next_job: # changing product
                    yield self.env.timeout(self.setup_time)

                if next_job=='A': 
                    # simulate 'getting' the material 
                    yield self.env.timeout(self.time_to_get_material)
                    # take the material for A from the prior buffer
                    yield self.env.material_m1_a.get(1) 
                    # processing the job
                    self.reward -= self.cost_for_machining * self.process_time_a[1]
                    yield self.env.timeout(self._apply_variability(self.process_time_a[1])) 
                    # place the finished good in next buffer
                    yield self.env.material_m2_a.put(1) 
                    #increase job counter by 1
                    self.finished_jobs_m2 += 1

                elif next_job=='B':
                    # simulate 'getting' the material 
                    yield self.env.timeout(self.time_to_get_material)
                    # take the material for B from the prior buffer
                    yield self.env.material_m1_b.get(1) 
                    # processing the job                    
                    self.reward -= self.cost_for_machining * self.process_time_b[1]
                    yield self.env.timeout(self._apply_variability(self.process_time_b[1])) 
                    # place the finished good in next buffer
                    yield self.env.material_m2_b.put(1) 
                    #increase job counter by 1
                    self.finished_jobs_m2 += 1

                else: 
                    raise ValueError('{} --> requested job is not in the list of allowed jobs [A, B]'.format(next_job))         

    def _run_machine_3(self):
        """ Runs the manufacutring programm of Machine 3 and needs to be called as a SimPy Process. """
        while True: 
            
            # Check if an order has to be executed
            jobs = len(self.fifo_list)
            # jobs == 0: no action yet, or only action 2
            # jobs == finished jobs: no more jobs available 
            if (jobs==0) or (jobs==self.finished_jobs_m3):
                yield self.env.timeout(self.time_step)
            else:
                # Check which order has to be executed
                next_job = self._get_next_job(self.finished_jobs_m3)

                # Check if a machine set-up is necessary due to product changes (A to B, or B to A)
                if jobs==1: # initial set-up required
                    yield self.env.timeout(self.setup_time)
                elif self.fifo_list[self.finished_jobs_m3-1] != next_job: # changing product
                    yield self.env.timeout(self.setup_time)

                if next_job=='A': 
                    # simulate 'getting' the material 
                    yield self.env.timeout(self.time_to_get_material)
                    # take the material for A from the prior buffer
                    yield self.env.material_m2_a.get(1) 
                    # processing the job
                    self.reward -= self.cost_for_machining * self.process_time_a[2]
                    yield self.env.timeout(self._apply_variability(self.process_time_a[2])) 
                    # place the finished good in next buffer
                    yield self.env.material_m3_a.put(1) 
                    #increase job counter by 1
                    self.finished_jobs_m3 += 1

                elif next_job=='B':
                    # simulate 'getting' the material 
                    yield self.env.timeout(self.time_to_get_material)
                    # take the material for B from the prior buffer
                    yield self.env.material_m2_b.get(1) 
                    # processing the job
                    self.reward -= self.cost_for_machining * self.process_time_b[2]
                    yield self.env.timeout(self._apply_variability(self.process_time_b[2])) 
                    # place the finished good in next buffer
                    yield self.env.material_m3_b.put(1) 
                    #increase job counter by 1
                    self.finished_jobs_m3 += 1

                else: 
                    raise ValueError('{} --> requested job is not in the list of allowed jobs [A, B]'.format(next_job))         

    def _run_machine_4(self):
        """ Runs the manufacutring programm of Machine 4 and needs to be called as a SimPy Process. """
        while True: 
            
            # Check if an order has to be executed
            jobs = len(self.fifo_list)
            # jobs == 0: no action yet, or only action 2
            # jobs == finished jobs: no more jobs available 
            if (jobs==0) or (jobs==self.finished_jobs_m4):
                yield self.env.timeout(self.time_step)
            else:
                # Check which order has to be executed
                next_job = self._get_next_job(self.finished_jobs_m4)

                # Check if a machine set-up is necessary due to product changes (A to B, or B to A)
                if jobs==1: # initial set-up required
                    yield self.env.timeout(self.setup_time)
                elif self.fifo_list[self.finished_jobs_m4-1] != next_job: # changing product
                    yield self.env.timeout(self.setup_time)

                if next_job=='A': 
                    # simulate 'getting' the material 
                    yield self.env.timeout(self.time_to_get_material)
                    # take the material for A from the prior buffer
                    yield self.env.material_m3_a.get(1) 
                    # processing the job
                    self.reward -= self.cost_for_machining * self.process_time_a[3]
                    yield self.env.timeout(self._apply_variability(self.process_time_a[3])) 
                    # place the finished good in next buffer
                    yield self.env.material_m4_a.put(1) 
                    #increase job counter by 1
                    self.finished_jobs_m4 += 1

                elif next_job=='B':
                    # simulate 'getting' the material 
                    yield self.env.timeout(self.time_to_get_material)
                    # take the material for B from the prior buffer
                    yield self.env.material_m3_b.get(1) 
                    # processing the job
                    self.reward -= self.cost_for_machining * self.process_time_b[3]
                    yield self.env.timeout(self._apply_variability(self.process_time_b[3])) 
                    # place the finished good in next buffer
                    yield self.env.material_m4_b.put(1) 
                    #increase job counter by 1
                    self.finished_jobs_m4 += 1

                else: 
                    raise ValueError('{} --> requested job is not in the list of allowed jobs [A, B]'.format(next_job))         

    def _run_machine_5(self):
        """ Runs the manufacutring programm of Machine 5 and needs to be called as a SimPy Process. """
        while True: 
            
            # Check if an order has to be executed
            jobs = len(self.fifo_list)
            # jobs == 0: no action yet, or only action 2
            # jobs == finished jobs: no more jobs available 
            if (jobs==0) or (jobs==self.finished_jobs_m5):
                yield self.env.timeout(self.time_step)
            else:
                # Check which order has to be executed
                next_job = self._get_next_job(self.finished_jobs_m5)

                # Check if a machine set-up is necessary due to product changes (A to B, or B to A)
                if jobs==1: # initial set-up required
                    yield self.env.timeout(self.setup_time)
                elif self.fifo_list[self.finished_jobs_m5-1] != next_job: # changing product
                    yield self.env.timeout(self.setup_time)

                if next_job=='A': 
                    # simulate 'getting' the material 
                    yield self.env.timeout(self.time_to_get_material)
                    # take the material for A from the prior buffer
                    yield self.env.material_m4_a.get(1) 
                    # processing the job
                    self.reward -= self.cost_for_machining * self.process_time_a[4]
                    yield self.env.timeout(self._apply_variability(self.process_time_a[4])) 
                    # place the finished good in next buffer
                    yield self.env.material_m5_a.put(1)
                    #increase job counter by 1
                    self.finished_jobs_m5 += 1
                    # 'sell' to customer and claim reward 
                    self.reward += self._calculate_sales_value('A')

                elif next_job=='B':
                    # simulate 'getting' the material 
                    yield self.env.timeout(self.time_to_get_material)
                    # take the material for B from the prior buffer
                    yield self.env.material_m4_b.get(1) 
                    # processing the job
                    self.reward -= self.cost_for_machining * self.process_time_b[4]
                    yield self.env.timeout(self._apply_variability(self.process_time_b[4])) 
                    # place the finished good in next buffer
                    yield self.env.material_m5_b.put(1) 
                    #increase job counter by 1
                    self.finished_jobs_m5 += 1
                    # 'sell' to customer and claim reward 
                    self.reward += self._calculate_sales_value('B')

                else: 
                    raise ValueError('{} --> requested job is not in the list of allowed jobs [A, B]'.format(next_job))         

    def _get_buffer_level_as_list(self):
        """ Returns a list of the current level of material in all twelve buffers. """
        return [self.env.raw_material_a.level, self.env.raw_material_b.level,
            self.env.material_m1_a.level, self.env.material_m1_b.level, 
            self.env.material_m2_a.level, self.env.material_m2_b.level, 
            self.env.material_m3_a.level, self.env.material_m3_b.level, 
            self.env.material_m4_a.level, self.env.material_m4_b.level, 
            self.env.material_m5_a.level, self.env.material_m5_b.level]

    def _get_observations(self):
        """ Returns the the current levels of all twelve buffers as observation list. """
        observations = self._get_buffer_level_as_list()
        return observations
                   
    def _islegal(self, action):
        """ Confirms that the selected action is in the action space, else an exception is raised. """
        # Check if action in action space
        if action not in self.actions:
            raise ValueError('{} --> requested action not in list of allowed actions [0, 1, 2'.format(action))

    def _release_raw_material(self, action):
        """ Adds new raw material to the initial buffer (only called in _perform_action). """
        if action==0: # release A
            yield self.env.raw_material_a.put(1)
            self.reward -= self.cost_for_raw_material
        elif action==1: # release B
            yield self.env.raw_material_b.put(1)
            self.reward -= self.cost_for_raw_material
        else:
            # do nothing for action==2
            pass

    def _perform_action(self, action):
        """ Changes the environment according to the selected action. """
        # Add action to all_actions
        self.all_actions += [action]
        # For 1 and 2: release new raw material and add job to fifo list
        if action == 0:
            # Add new job to fifo list
            self.fifo_list += ['A']
            # Release raw material A to buffer
            self.env.process(self._release_raw_material(action))
        elif action==1: 
            # Add new job to fifo list
            self.fifo_list += ['B']
            # Release raw material B to buffer
            self.env.process(self._release_raw_material(action))
        elif action==2: 
            # do nothing for action==2
            self.reward -= 0


    def reset(self):
        """ Initializes/resets the simulation after completing a run over a full shift (t=480). """ 

        # Initialize simpy environemnt
        self.env = simpy.Environment()

        # Set up buffer with raw material A and B, before Machine 1
        self.env.raw_material_a = simpy.Container(self.env, capacity=self.buffer_capacity, init=self.initial_buffer)
        self.env.raw_material_b = simpy.Container(self.env, capacity=self.buffer_capacity, init=self.initial_buffer)

        # Buffer with semi-finished goods A and B, after Machine 1
        self.env.material_m1_a = simpy.Container(self.env, capacity=self.buffer_capacity, init=self.initial_buffer)
        self.env.material_m1_b = simpy.Container(self.env, capacity=self.buffer_capacity, init=self.initial_buffer)

        # Buffer with semi-finished goods A and B, after Machine 2
        self.env.material_m2_a = simpy.Container(self.env, capacity=self.buffer_capacity, init=self.initial_buffer)
        self.env.material_m2_b = simpy.Container(self.env, capacity=self.buffer_capacity, init=self.initial_buffer)

        # Buffer with semi-finished goods A and B, after Machine 3
        self.env.material_m3_a = simpy.Container(self.env, capacity=self.buffer_capacity, init=self.initial_buffer)
        self.env.material_m3_b = simpy.Container(self.env, capacity=self.buffer_capacity, init=self.initial_buffer)

        # Buffer with semi-finished goods A and B, after Machine 4
        self.env.material_m4_a = simpy.Container(self.env, capacity=self.buffer_capacity, init=self.initial_buffer)
        self.env.material_m4_b = simpy.Container(self.env, capacity=self.buffer_capacity, init=self.initial_buffer)

        # Buffer with semi-finished goods A and B, after Machine 5
        self.env.material_m5_a = simpy.Container(self.env, capacity=self.buffer_capacity, init=self.initial_buffer)
        self.env.material_m5_b = simpy.Container(self.env, capacity=self.buffer_capacity, init=self.initial_buffer)

        # Reset counter for all finished jobs
        self.finished_jobs_m1 = 0
        self.finished_jobs_m2 = 0
        self.finished_jobs_m3 = 0
        self.finished_jobs_m4 = 0
        self.finished_jobs_m5 = 0

        # Reset counter for reward according to product mix
        self.consecutive_production_of_a = 0 
        self.consecutive_production_of_b = 0
        self.consecutive_idle_states = 0 

        # Reset simulation parameters for next run
        self.next_time_stop = 0
        self.fifo_list = []
        self.all_actions = []
        self.reward = self.initial_reward

        # Run all machines
        self.env.process(self._run_machine_1())
        self.env.process(self._run_machine_2())
        self.env.process(self._run_machine_3())
        self.env.process(self._run_machine_4())
        self.env.process(self._run_machine_5())

        # Return starting state observations
        observations = self._get_observations()
        return observations

    def step(self, action): 
        """ Executes the next simulation step (acc. to self.time_step) and performes the choosen action. """

        # Reset reward to initial reward value (neg. term for DQN)
        self.reward = self.initial_reward # + self.env.material_m5_a.level + self.env.material_m5_b.level

        # Proceed if the selected action is legal else raise an exception
        self._islegal(action)

        # Adjust env according to selected action (e.g. release raw material and add job to fifo list)
        self._perform_action(action)

        if False: #(self.env.now in [0, 240, 479]):
            print('t=' + str(self.env.now) + ' ' + str(self._get_observations()) + ' ' + str(self.reward))

        # Make a step in the simulation
        self.next_time_stop += self.time_step
        self.env.run(until=self.next_time_stop)

        # Get new observations 
        observations = self._get_observations()

        # Determine the cost for storing all raw material and semi finished goods
        # Excluding costs for finished goods after M5 (--> customer)
        cost_for_storage = sum(self._get_buffer_level_as_list()[:-2]) * self.cost_for_storage
        
        # Punish the agent for selecting multiple idle states in a row 
        if action!=2:
            self.consecutive_idle_states = 0
        else: 
            self.consecutive_idle_states += 1

        # Adjust reward for storage costs and return as new reward 
        reward = self.reward - cost_for_storage - self.consecutive_idle_states

        # Check if terminal state was reached (t=simulation_duration)
        terminal = True if (self.env.now >= self.simulation_duration) else False

        # Information is an empty dictionary, used to be compatble with OpenAI Gym
        info = dict()  

        # Return tuple of observations, reward, terminal, info
        return (observations, reward, terminal, info)
