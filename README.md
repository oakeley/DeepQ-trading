# DeepQ-trading
Code looks at historical values to make an estimate for which shares (and how many) to add to the portfolio. It then runs 100 episodes of trading trying to maximise the return by seeking the maximum reward.

## Stock Universe and Data Management:


The code works with 20 Swiss stocks (indicated by the .SW suffix, downloaded from the SIX Swiss Exchange)
It includes major companies like Nestlé (NESN.SW), Novartis (NOVN.SW), and UBS (UBSG.SW)
The system loads historical stock data either from local (cached) CSV files or downloads it using the yfinance library if no files exist.


## Trading Environment (TradingEnvironment class):


Simulates a trading environment with an initial balance of 1,000,000 (CHF)
Uses a sliding window of 20 days to make decisions
Includes realistic trading features like:

Transaction costs (fixed cost of 1 CHF + 0.1% variable cost)
Portfolio tracking
Risk-adjusted returns calculation




## Neural Network Architecture (DuelingDQN class):


Implements a Dueling Deep Q-Network, which is a sophisticated type of reinforcement learning architecture
Separates the value and advantage streams, which helps in better estimating action values
Uses multiple layers to process the market state information


## Trading Agent (DRLTrader class):


Implements an epsilon-greedy strategy for action selection (balancing exploration and exploitation)
Can perform three actions for each stock: buy, hold, or sell
Uses experience replay (via ReplayBuffer) to store and learn from past trading decisions
Includes both a policy network and a target network for stable learning
![image](https://github.com/user-attachments/assets/d75fe0c1-034f-43da-bf05-55b547e1084d)
