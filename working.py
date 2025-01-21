import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Define the stock universe
SWISS_STOCKS = [
    'ABBN.SW', 'ADEN.SW', 'ALC.SW', 'CFR.SW', 'GEBN.SW',
    'GIVN.SW', 'HOLN.SW', 'KNIN.SW', 'LONN.SW', 'NESN.SW',
    'NOVN.SW', 'PGHN.SW', 'ROG.SW', 'SGSN.SW', 'SIKA.SW',
    'SLHN.SW', 'SOON.SW', 'UBSG.SW', 'UHR.SW', 'ZURN.SW'
]

class TradingEnvironment:
    def __init__(self, stocks=SWISS_STOCKS, initial_balance=1000000, window_size=20):
        self.stocks = stocks
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.data = {}

        # Try to configure dynamic plots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12))
        self.portfolio_value_history = [] # Portfolio value for each episode
        self.reward_history = [] # Total reward for each episode
        self.total_reward = 0 # Total reward for the current episode
        self.buy_trades = 0
        self.sell_trades = 0
        self.trade_history = []
        self.current_episode = 1 # Current episode number, they start from 1
        
        if not self._load_data():
            raise RuntimeError("Failed to initialize trading environment due to data loading errors")
            
        self.current_step = self.window_size
        self.max_steps = min(len(df) for df in self.data.values()) - window_size

        # Analyze historical data and build initial portfolio
        self._analyze_historical_data()
        self._build_initial_portfolio()

        self.reset()

    def _analyze_historical_data(self):
        self.historical_performance = {}
        for stock in self.stocks:
            prices = self.data[stock]['Close'][:self.window_size]
            returns = prices.pct_change().dropna()
            mean_return = returns.mean()
            self.historical_performance[stock] = mean_return

    def _build_initial_portfolio(self):
        sorted_stocks = sorted(self.historical_performance, key=self.historical_performance.get, reverse=True)
        self.positions = {stock: 0 for stock in self.stocks}
        self.balance = self.initial_balance

        for stock in sorted_stocks:
            price = self.data[stock]['Close'][self.window_size]
            max_shares = self.balance // (price + self._calculate_transaction_costs(price))
            if max_shares > 0:
                self.positions[stock] = max_shares
                self.balance -= max_shares * price + self._calculate_transaction_costs(max_shares * price)
                if self.balance <= 0:
                    break

    def _calculate_transaction_costs(self, value):
        # Transaction cost calculation based on the transaction value
        return max(1, 0.001 * value)  # Example: 0.1% of the transaction value with a minimum cost of 1 unit


    def _load_data(self):
        # Load data from files or download if not available
        try:
            os.makedirs('stock_data', exist_ok=True)
        except Exception as e:
            print(f"Error creating directory: {e}")
            return False

        success = True
        for stock in self.stocks:
            file_path = f'stock_data/{stock.replace(".", "_")}.csv'

            try:
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    print(f"Loading existing data for {stock}...")
                    # Read data and save column titles
                    df = pd.read_csv(file_path)
                    col = df.columns.tolist()
                    # First column is incorrectly called 'Price' even though it is a date
                    col[0] = 'Date'
                    # Load again without the first two rows
                    df = pd.read_csv(file_path, skiprows=[0, 1])
                    # Reapply the column titles
                    df.columns = col
                    
                    # Convert Date column to datetime and set as index
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    
                    # Convert all numeric columns directly
                    numeric_columns = ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Ensure required columns exist
                    required_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        print(f"Missing required columns for {stock}: {missing_columns}")
                        # If Close is missing but Price exists, use Price as Close
                        if 'Close' in missing_columns and 'Price' in df.columns:
                            df['Close'] = df['Price']
                            missing_columns.remove('Close')
                        if missing_columns:  # If still missing columns
                            success = False
                            continue
                    
                    # Convert string values to numeric
                    for col in required_columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception as e:
                            print(f"Error converting {col} column to numeric for {stock}: {e}")
                    
                    # Use ffill() and bfill() instead of deprecated fillna(method=...)
                    df = df.ffill().bfill()
                    
                    # Verify data quality
                    if df['Close'].isnull().any():
                        print(f"Missing values in Close prices for {stock} after filling")
                        success = False
                        continue
                        
                    if len(df) < self.window_size:
                        print(f"Insufficient data for {stock}: {len(df)} rows")
                        success = False
                        continue
                        
                    # Add date index if not present
                    if 'Date' not in df.columns:
                        df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='B')
                    else:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    
                    self.data[stock] = df
                    
                else:
                    print(f"Downloading data for {stock}...")
                    df = yf.download(stock, start="2020-01-01", end="2024-01-01")
                    if len(df) == 0:
                        print(f"No data downloaded for {stock}")
                        success = False
                        continue
                    try:
                        df.to_csv(file_path, index=True)
                        self.data[stock] = df
                    except Exception as e:
                        print(f"Error saving data for {stock}: {e}")
                        success = False
                        continue
                        
            except Exception as e:
                print(f"Error processing {stock}: {e}")
                success = False
                
        # Final verification
        if success:
            print(f"Successfully loaded data for {len(self.data)} stocks")
            # Print first few rows of first stock for verification
            first_stock = next(iter(self.data.values()))
            print("\nSample of loaded data:")
            print(first_stock.head())
            print("\nColumns:", first_stock.columns.tolist())
        else:
            print("Failed to load all required stock data")
            
        return success and len(self.data) == len(self.stocks)

    def reset(self):
        self.balance = self.initial_balance
        self.positions = {stock: 0 for stock in self.stocks}
        self.portfolio_value_history = [self.initial_balance]
        self.current_step = self.window_size
        return self._get_state()

    def _calculate_transaction_costs(self, value):
        fixed_cost = 1.0  # CHF
        variable_cost_percentage = 0.001  # 0.1%
        return fixed_cost + value * variable_cost_percentage

    def _get_state(self):
        state = []
        for stock in self.stocks:
            df = self.data[stock]
            window = df.iloc[self.current_step - self.window_size:self.current_step]

            # Price features
            try:
                # Calculate returns and ensure they are finite values
                returns = window['Close'].pct_change().fillna(0).values[1:]
                returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0).tolist()
                
                # Calculate normalized prices
                normalized_prices = (window['Close'] / window['Close'].iloc[0]).values
                normalized_prices = np.nan_to_num(normalized_prices, nan=1.0, posinf=1.0, neginf=1.0).tolist()
            except Exception as e:
                print(f"Error calculating returns/prices: {e}")
                returns = [0.0] * (self.window_size - 1)
                normalized_prices = [1.0] * self.window_size

            # Verify and fix the length of features
            if len(returns) != self.window_size - 1:
                returns = [0.0] * (self.window_size - 1)
            if len(normalized_prices) != self.window_size:
                normalized_prices = [1.0] * self.window_size

            # Technical indicators
            rsi = float(self._calculate_rsi(window['Close']))
            # Calculate volatility as standard deviation of returns
            returns_array = np.array(returns)
            volatility = float(np.std(returns_array)) if len(returns) > 0 else 0.0
            position = float(self.positions[stock])

            # Add features one by one to ensure they're all floats
            state.extend([float(r) for r in returns])
            state.extend([float(p) for p in normalized_prices])
            state.extend([rsi, volatility, position])

        # Add portfolio state
        portfolio_value = self._get_portfolio_value()
        state.append(float(self.balance / self.initial_balance))
        state.append(float(portfolio_value / self.initial_balance))

        # Convert to numpy array and handle any remaining NaN values
        state_array = np.array(state, dtype=np.float32)
        
        # Debug print to verify state shape
        if len(state) != (self.window_size - 1 + self.window_size + 3) * len(self.stocks) + 2:
            print(f"Warning: Unexpected state length. Got {len(state)}, expected {(self.window_size - 1 + self.window_size + 3) * len(self.stocks) + 2}")
        
        return np.nan_to_num(state_array, nan=0.0)

    def _calculate_rsi(self, prices, periods=14):
        returns = prices.diff()
        gains = returns.clip(lower=0)
        losses = -returns.clip(upper=0)

        avg_gain = gains.rolling(window=periods, min_periods=1).mean()
        avg_loss = losses.rolling(window=periods, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _get_portfolio_value(self):
        value = self.balance
        for stock in self.stocks:
            price = self.data[stock]['Close'].iloc[self.current_step]
            value += self.positions[stock] * price
        return value

    def step(self, actions):
        # Execute one step in the environment
        if not isinstance(actions, dict) or not all(stock in actions for stock in self.stocks):
            raise ValueError("Actions must be provided for all stocks")
            
        prev_value = self._get_portfolio_value()

        # Execute trades
        for stock, (action, amount) in actions.items():
            price = self.data[stock]['Close'].iloc[self.current_step]

            if action == 0:  # Sell
                max_sell = self.positions[stock]
                shares_to_sell = int(max_sell * amount)
                if shares_to_sell > 0:
                    value = shares_to_sell * price
                    transaction_cost = self._calculate_transaction_costs(value)
                    self.balance += value - transaction_cost
                    self.positions[stock] -= shares_to_sell
                    self.sell_trades += 1  # Track sell trades

            elif action == 2:  # Buy
                max_buy = self.balance / price
                shares_to_buy = int(max_buy * amount)
                if shares_to_buy > 0:
                    value = shares_to_buy * price
                    transaction_cost = self._calculate_transaction_costs(value)
                    if value + transaction_cost <= self.balance:
                        self.balance -= value + transaction_cost
                        self.positions[stock] += shares_to_buy
                        self.buy_trades += 1  # Track buy trades

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Calculate reward
        new_value = self._get_portfolio_value()
        self.portfolio_value_history.append(new_value)
        
        pvc = (new_value - prev_value) / prev_value

        # Risk-adjusted return
        recent_returns = np.diff(self.portfolio_value_history[-21:]) / self.portfolio_value_history[-21:-1]
        volatility = recent_returns.std() if len(recent_returns) > 1 else 0
        risk_free_rate = 0.0002  # Daily risk-free rate
        rar = (pvc - risk_free_rate) / (volatility + 1e-6)

        # Combined reward
        reward = pvc - 0.1 * volatility + 0.5 * rar
        self.total_reward += reward

        # Update plot at the end of the episode
        if done:
            self.reward_history.append(self.total_reward)
            self.trade_history.append((self.buy_trades, self.sell_trades))
            self.total_reward = 0  # Reset total reward for the next episode
            self.buy_trades = 0  # Reset buy trades for the next episode
            self.sell_trades = 0  # Reset sell trades for the next episode

            clear_output(wait=True)
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()

            self.ax1.plot(self.portfolio_value_history, label='Portfolio Value')
            self.ax1.set_title(f'Portfolio Value Change (Episode: {self.current_episode})')
            self.ax1.set_xlabel('Steps')
            self.ax1.set_ylabel('Value')
            self.ax1.legend()

            self.ax2.plot(self.reward_history, label='Total Reward', color='orange')
            self.ax2.set_title('Total Reward Over Episodes')
            self.ax2.set_xlabel('Episodes')
            self.ax2.set_ylabel('Total Reward')
            self.ax2.legend()
            self.ax2.set_xticks(range(len(self.reward_history)))
            # We use episode numbers as x-ticks but they need to be integers and start from 1
            self.ax2.set_xticklabels([str(np.int32(i + 1)) for i in range(len(self.reward_history))])

            buy_trades, sell_trades = zip(*self.trade_history)
            self.ax3.plot(buy_trades, label='Buy Trades', color='green')
            self.ax3.plot(sell_trades, label='Sell Trades', color='red')
            self.ax3.set_title('Number of Trades Over Episodes')
            self.ax3.set_xlabel('Episodes')
            self.ax3.set_ylabel('Number of Trades')
            self.ax3.legend()
            self.ax3.set_xticks(range(len(self.trade_history)))
            # We use episode numbers as x-ticks but they need to be integers and start from 1
            self.ax3.set_xticklabels([str(np.int32(i + 1)) for i in range(len(self.trade_history))])

            plt.tight_layout()
            plt.pause(0.001)

            self.current_episode += 1  # Increment the episode number

        return self._get_state(), reward, done

class DuelingDQN(nn.Module): # Dueling Deep Q-Network
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.state_size = state_size # Number of features in the state
        self.action_size = action_size # Number of actions
        
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 256), # Input layer
            nn.ReLU(), # Activation function
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.advantage_stream = nn.Sequential( # Advantage stream
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        shared = self.shared_layers(state)
        advantage = self.advantage_stream(shared)
        value = self.value_stream(shared)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # Maximum number of transitions to store in the buffer
        self.memory = deque(maxlen=capacity) # Internal memory (deque)
        self.Transition = namedtuple('Transition', 
                                   ('state', 'actions_dict', 'reward', 'next_state', 'done'))
        
    def push(self, state, actions_dict, reward, next_state, done):
        # Store a transition
        self.memory.append(self.Transition(state, actions_dict, reward, next_state, done))
        
    def sample(self, batch_size):
        # Sample a batch of transitions
        if len(self.memory) < batch_size:
            return None
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Deep Reinforcement Learning Trader
class DRLTrader: 
    def __init__(self, state_size, num_stocks, learning_rate=0.0003):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_stocks = num_stocks # Number of stocks to trade
        self.actions_per_stock = 3  # sell, hold, buy
        
        # Network handles all stocks simultaneously
        # Policy network: This network is used to select actions based on the current state.
        self.policy_net = DuelingDQN(state_size, self.actions_per_stock * num_stocks).to(self.device) # Policy network
        # Target network: This network is used to compute the target Q-values for training the policy network.
        # It is a copy of the policy network and is updated less frequently to provide stable targets.
        self.target_net = DuelingDQN(state_size, self.actions_per_stock * num_stocks).to(self.device) # Target network
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Initialize target network with policy network weights
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(100000) # Replay buffer to store transitions
        
        self.batch_size = 256
        self.gamma = 0.99  # Discount factor for future rewards
        self.tau = 0.005  # Soft update parameter for target network
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration probability
        
    def select_actions(self, state):
        # Select actions for all stocks simultaneously
        if random.random() < self.epsilon:
            return {stock: (random.randint(0, 2), random.random()) 
                   for stock in SWISS_STOCKS}
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy().reshape(-1, self.actions_per_stock)
            actions = {}
            
            for i, stock in enumerate(SWISS_STOCKS):
                action = q_values[i].argmax()
                actions[stock] = (action, random.random())
                
            return actions

    def train(self):
        # Train the network on a batch of transitions
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return
        
        # Convert lists to numpy arrays before creating tensors
        states = np.array([t.state for t in batch])
        next_states = np.array([t.next_state for t in batch])
        rewards = np.array([t.reward for t in batch])
        dones = np.array([t.done for t in batch])
        
        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Convert actions dictionary to tensor
        actions_list = []
        for transition in batch:
            stock_actions = []
            for stock in SWISS_STOCKS:
                stock_actions.append(transition.actions_dict[stock][0])  # Get action type only
            actions_list.append(stock_actions)
        actions = np.array(actions_list)  # Convert to numpy array first
        actions = torch.LongTensor(actions).to(self.device)
        
        # Calculate current Q values
        current_q_values = self.policy_net(states)
        current_q_values = current_q_values.reshape(-1, self.num_stocks, self.actions_per_stock)
        current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        
        # Calculate next Q values
        next_q_values = self.target_net(next_states)
        next_q_values = next_q_values.reshape(-1, self.num_stocks, self.actions_per_stock)
        next_q_values = next_q_values.max(2)[0]
        
        # Calculate expected Q values
        expected_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Calculate loss
        loss = nn.HuberLoss()(current_q_values, expected_q_values.detach())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                            self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def main():
    # Initialize environment and agent
    env = TradingEnvironment()
    state_size = len(env.reset())
    num_stocks = len(SWISS_STOCKS)
    
    trader = DRLTrader(state_size, num_stocks)
    
    # Training loop
    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            actions = trader.select_actions(state)
            next_state, reward, done = env.step(actions)
            trader.memory.push(state, actions, reward, next_state, done)
            
            if len(trader.memory) >= trader.batch_size:
                trader.train()
            
            state = next_state
            total_reward += reward
        
        portfolio_value = env._get_portfolio_value()
        print(f"Episode {episode + 1}/{num_episodes}, "
              f"Total Reward: {total_reward:.2f}, "
              f"Portfolio Value: {portfolio_value:.2f}, "
              f"Epsilon: {trader.epsilon:.3f}")

        # Optional: Save model periodically
        if (episode + 1) % 10 == 0:
            torch.save({
                'episode': episode,
                'policy_net_state_dict': trader.policy_net.state_dict(),
                'target_net_state_dict': trader.target_net.state_dict(),
                'optimizer_state_dict': trader.optimizer.state_dict(),
                'epsilon': trader.epsilon
            }, f'model_checkpoint_{episode + 1}.pt')

if __name__ == "__main__":
    main()
