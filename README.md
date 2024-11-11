# A Reinforcement Learning Optimization Framework for UAV Positioning Information Assisted IoT Data Collection

Unmanned Aerial Vehicles (UAVs) have attracted
widespread attention in the field of assisted data collection
and localization due to their flexibility, mobility, and ease of
deployment. In this paper, we put forth a framework for UAVs
to leverage real-time localization data to assist data collection
in a low-latency-sensitive, energy-efficient sensor network. This
framework addresses the shortcomings of inadequate infrastructure coverage, the inherent unpredictability and ambiguity
of the data collection process, and the challenges posed by
sensor locations in the Internet of Things (IoT). The proposed framework markedly enhances system performance by
leveraging real-time localization data to assist data collection.
Specifically, we optimize the traditional algorithms in terms of
model construction (e.g., the PER DQN algorithm, the Dueling
DQN algorithm, etc.), data collection strategy, etc., to reduce
the computational complexity and resource consumption of the
UAV and assist the UAV in obtaining the optimal strategy for
trajectory planning, in response to a series of problems, such
as the lower processing efficiency and slower convergence speed
of the traditional algorithms. Extensive simulation results show
that our proposed solution improves the average data collection
performance of the system by 17% and reduces the convergence
time to 83.3% compared to traditional methods.
![](https://github.com/wangsh386/NEW_Dueling-Per_DQN/blob/main/images/overview.png)

## 1. Requirements

ipython==8.12.3
matplotlib==3.8.0
numpy==1.26.4
torch==2.1.0



## 2. Usage

Run the following command:

training：

```bash
python DQN.py
```

testing：

```bash
python watch_uav.py
```
### Convergence comparison results of six reinforcement learning algorithms.
![](https://github.com/wangsh386/NEW_Dueling-Per_DQN/blob/main/images/compare.png)
