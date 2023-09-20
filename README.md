# Reinforcement Learning for Production Program Planning: ASIM 2021
This repository contains results of following conference publication: 


## Publication
#### Title
An Approach for Deep Reinforcement Learning for Production Program Planning in Value Streams

#### Authors
- Nikolai West <sup> [ORCID](https://orcid.org/0000-0002-3657-0211) </sup>
- Florian Hoffmann <sup> [ORCID](https://orcid.org/0000-0002-0276-4026) </sup>
- Lukas Schulte <sup> [ORCID](https://orcid.org/0000-0002-0613-7927) </sup>
- Victor Hernandez Moreno <sup> [ORCID](https://orcid.org/0000-0002-6038-6959) </sup>
- Jochen Deuse <sup> [ORCID](https://orcid.org/0000-0003-4066-4357) </sup>

#### Abstract 
The application of Reinforcement Learning (RL) methods offers a potential for improvement in operational Production Program Planning. Numerous influences and domain-specific practices characterize the multi-dimensional planning paradigm. RL can support human planning personnel in the determination of optimal production parameters. This requires a suitable abstraction of the overall system by means of simulation and subsequent optimization by a self-learning agent. In this paper, the authors present an application example for sequence planning using RL. The case study includes a discrete-event simulation built with SimPy that is trained by a Duelling Deep-Q-Network implemented in PyTorch. Finally, the suitability of two reward functions is discussed. The authors fully provide the case study via GitHub.

#### Conference 
2021 ASIM 2nd Simulation in Production and Logistics (ASIM 2021)

#### Status
- Published ([available](http://www.asim-fachtagung-spl.de/asim2021/papers/Proof_172.pdf))


## Contents
The repository contains two main files to perform the proposed approach for deep reinforcement learning for production program planning in manufacturing value streams. 

#### 1. Simulation of a simple manufacturing system ([factory_simulation.py](https://github.com/nikolaiwest/2021-reinforcement-learning-asim/blob/main/factory_simulation.py))
Contains Python code for a simple manufacturing simulation. This simulation models a factory with five stations and two products (A and B). It is implemented using SimPy and provides an interface that resembles an OpenAI Gym environment, making it suitable for Reinforcement Learning applications.

![Layout of the simulation](https://github.com/nikolaiwest/2021-reinforcement-learning-asim/blob/main/layout.png)


#### 2. Deep reinforcement learning agent using DDQN ([factory_agent.py](https://github.com/nikolaiwest/2021-reinforcement-learning-asim/blob/main/factory_agent.py))
Contains Python code that defines a reinforcement learning agent for optimizing manufacturing processes. The agent uses Q-learning to learn the best actions to take in a simulated factory environment, balancing exploration and exploitation through an epsilon-greedy strategy. It also includes functions for training the agent and making decisions to maximize rewards and improve manufacturing efficiency. The design of the simulation is choosen as to allow one ideal scenario, outlined below. It is the agents job to identify such a pattern using the DDQN.

![Scheduling task for the agent](https://github.com/nikolaiwest/2021-reinforcement-learning-asim/blob/main/scheduling.png)


## Results
All results of the training are made available in the repsository ([/results](https://github.com/nikolaiwest/2021-reinforcement-learning-asim/tree/main/results)). The folder contains data and plots for both reward functions (RF1 and RF2) as outlined in the [paper](http://www.asim-fachtagung-spl.de/asim2021/papers/Proof_172.pdf).  


## Usage
1. **Clone the Repository:** Clone this repository to your local machine 
2. **Install Dependencies:** Set up a new env using [requirements.txt](https://github.com/nikolaiwest/2021-reinforcement-learning-asim/blob/main/requirements.txt)
3. **Run the Project:** You can now run the project. For example:

- To run the simulation, use the following command:
  ```
  python factory_simulation.py
  ```

- To train the agent, execute the training script:
  ```
  python train_agent.py
  ```
4. **Explore the Results:** After running, you can explore the generated results, plots, or trained models based on the parameters you have set for the simulation (or agent).

## Contributing
We welcome contributions to this repository. If you have a feature request, bug report, or proposal, please open an issue. If you wish to contribute code, please open a pull request.