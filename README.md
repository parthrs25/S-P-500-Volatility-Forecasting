# S&P 500 Volatility Forecasting

Predicting market volatility is one of the most challenging problems in financial analysis. This project explores multiple approaches to forecast S&P 500 volatility using traditional econometric models and modern deep learning techniques.

## Why This Project?

Market volatility isn't just an academic curiosity—it's a critical measure that affects investment decisions, risk management strategies, and portfolio allocation. The VIX index, often called the "fear gauge," provides a market-based estimate of expected volatility. But can we forecast it accurately using historical data?

I wanted to find out by comparing different modeling approaches, from classical EWMA (Exponentially Weighted Moving Average) to sophisticated neural networks. The results were fascinating and sometimes surprising.

## The Dataset

The analysis covers nearly 15 years of daily market data from May 2005 to December 2019, including:

- **S&P 500 Index (^GSPC)**: Daily open, high, low, close prices and trading volume
- **VIX Index (^VIX)**: The CBOE Volatility Index, representing market expectations of 30-day volatility
- **Derived Features**: Log returns, RSI (Relative Strength Index), moving averages, and various technical indicators

After cleaning and feature engineering, the dataset contains **3,676 observations** with **70 engineered features**, carefully designed to avoid multicollinearity issues that plagued earlier iterations.

## Methodology

### Data Preparation

The raw OHLC (Open-High-Low-Close) data presented a significant challenge: perfect correlations between price columns. To solve this, I created a cleaned dataset that:

- Retains only the Close price to avoid redundancy
- Adds meaningful derived features like log returns and volatility ratios
- Normalizes technical indicators
- Maintains temporal relationships crucial for time series forecasting

### Train/Validation/Test Split

I used a strict chronological 80/10/10 split:
- **Training**: 2,940 observations (80%)
- **Validation**: 368 observations (10%)
- **Test**: 368 observations (10%)

This ensures no data leakage—the model never sees future data during training, mimicking real-world forecasting scenarios.

### Models Compared

#### 1. EWMA (Exponentially Weighted Moving Average)
The baseline approach. EWMA gives more weight to recent observations, making it responsive to changing market conditions.

**Best Configuration**: Lambda = 0.94
- **Test RMSE**: 13.97
- **Test R²**: 0.52
- **Directional Accuracy**: Not applicable (simple smoothing)

#### 2. ARCH/GARCH Models
These econometric models explicitly capture volatility clustering—the tendency of large price changes to cluster together.

**Best Model**: GARCH(1,1)
- **Test RMSE**: 10.96
- **Test R²**: 0.69
- **Interpretation**: Successfully captures volatility persistence

#### 3. Recurrent Neural Networks (RNN)

After hyperparameter optimization, the optimal RNN configuration emerged:

**Architecture**: 
- Single layer with 16 units
- Tanh activation
- 20% dropout
- Sequence length: 15 days

**Performance**:
- **Train RMSE**: 2.93
- **Test RMSE**: 10.96
- **Test R²**: 0.69
- **Directional Accuracy**: 71.1%

The RNN showed strong performance with good generalization, though there's evidence of some overfitting (274.5% RMSE increase from train to test).

#### 4. Gated Recurrent Units (GRU)

GRUs are computationally more efficient than LSTMs and often perform comparably. Multiple configurations were tested.

**Best Model**: GRU_Medium_v2
- **Architecture**: 32 units, 2 layers, tanh activation
- **Test RMSE**: 9.07
- **Test R²**: 0.79
- **Training Time**: 29.2 seconds

**Runner-up**: GRU_Simple_v2
- **Test RMSE**: 9.61
- **Test R²**: 0.76

Interestingly, the initial GRU implementations produced suspiciously flat predictions—a common issue when models fail to learn temporal patterns. After careful debugging and using the proven RNN architecture as a template, the corrected GRU models showed substantial improvement.

#### 5. Long Short-Term Memory (LSTM)

LSTM networks are designed to handle long-term dependencies in sequential data through their sophisticated gating mechanisms.

**Best Model**: LSTM_Simple
- **Architecture**: 16 units, single layer
- **Test RMSE**: 9.51
- **Test R²**: 0.77
- **Training Time**: 19.3 seconds

**Strong Performer**: LSTM_Medium
- **Test RMSE**: 9.35
- **Test R²**: 0.77

## Key Results

### Model Performance Comparison

| Model | Test RMSE | Test R² | Directional Accuracy | Training Time |
|-------|-----------|---------|---------------------|---------------|
| **GRU_Medium_v2** | **9.07** | **0.79** | - | 29.2s |
| LSTM_Medium | 9.35 | 0.77 | - | 30.0s |
| LSTM_Simple | 9.51 | 0.77 | - | 19.3s |
| GRU_Simple_v2 | 9.61 | 0.76 | - | 16.3s |
| GARCH(1,1) | 10.96 | 0.69 | - | Fast |
| RNN_Optimal | 10.96 | 0.69 | 71.1% | 15-20s |
| EWMA (λ=0.94) | 13.97 | 0.52 | - | Instant |

### What I Learned

1. **Deep Learning Wins, But Not by Much**: The best GRU model achieved an R² of 0.79, compared to GARCH's 0.69. That's meaningful, but not revolutionary.

2. **Simpler Can Be Better**: The single-layer LSTM_Simple performed nearly as well as deeper architectures, training 60% faster. In production, this matters.

3. **The Debugging Journey**: The initial GRU models produced flat predictions (std dev ~1.3, while actual volatility ranged from 5 to 123). This taught me an important lesson about architecture choices and activation functions. The "corrected" GRU models, following the proven RNN pattern, showed dramatic improvement.

4. **Directional Accuracy Matters**: The RNN achieved 71.1% directional accuracy. For trading strategies, predicting whether volatility will go up or down can be more valuable than predicting the exact magnitude.

5. **Multicollinearity Is Real**: Early versions failed because of perfect correlations between OHLC prices. Feature engineering wasn't optional—it was essential.

## Visualizations

The project includes comprehensive visualizations showing:
- Time series plots comparing predictions vs actual volatility across train/val/test periods
- Residual analysis to check for patterns in prediction errors
- Correlation heatmaps before and after feature engineering
- Model comparison charts

All plots are saved in the `model_plots/` directory.

## Technical Details

### Requirements
```python
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow >= 2.20.0
arch  # for ARCH/GARCH models
yfinance  # for data fetching
```

### Running the Analysis

1. **Clone the repository**:
```bash
git clone https://github.com/ashwini0008/S-P-500-VOLATILITY-FORECASTING.git
cd S-P-500-VOLATILITY-FORECASTING
```

2. **Open the Jupyter notebook**:
```bash
jupyter notebook VolatilityModelling.ipynb
```

3. **Run all cells**: The notebook is designed to be executed sequentially. It will:
   - Fetch and prepare data
   - Handle multicollinearity issues
   - Train all models
   - Generate comprehensive visualizations
   - Save results and plots

### Project Structure
```
├── README.md                      # This file
├── VolatilityModelling.ipynb     # Main analysis notebook
└── model_plots/                  # Generated visualizations
    ├── EWMA_*.png
    ├── GARCH_*.png
    ├── RNN_*.png
    ├── GRU_*.png
    └── LSTM_*.png
```

## Limitations and Future Work

### Current Limitations
- **Overfitting**: Some models show significant train-test performance gaps
- **Data Period**: Analysis ends in 2019; COVID-era volatility isn't captured
- **Feature Selection**: While engineered features helped, more sophisticated feature selection could improve results
- **Hyperparameter Search**: The "fast" HPO was time-constrained; more thorough search might yield better configurations

### Future Improvements
- **Attention Mechanisms**: Transformer-based architectures could better capture long-range dependencies
- **Ensemble Methods**: Combining GARCH with neural networks might leverage strengths of both
- **External Data**: Incorporating news sentiment, options data, or macroeconomic indicators
- **Walk-Forward Optimization**: Retraining periodically as new data arrives
- **Risk-Adjusted Metrics**: Beyond RMSE, evaluate using Sharpe ratios or other financial metrics

## Conclusion

Forecasting volatility is hard. Really hard. But it's not impossible.

This project demonstrates that modern deep learning methods, particularly GRU networks, can outperform traditional econometric models in forecasting S&P 500 volatility. The best GRU model achieved an R² of 0.79 on unseen test data—a solid result for such a noisy prediction task.

However, the margin of improvement over GARCH wasn't enormous, and simpler models trained much faster. In a production environment, the choice between a GARCH model and a deep learning model would depend on your specific requirements: Do you need the extra 10% accuracy? Can you afford the computational cost? How often do you need to retrain?

The journey also highlighted the importance of careful data preparation, particularly handling multicollinearity, and the value of rigorous model debugging. The "horizontal prediction" issue in early GRU implementations was a humbling reminder that deep learning isn't magic—architecture choices and hyperparameters matter immensely.

---

**Project by**: Ashwini  
**Data Source**: Yahoo Finance (yfinance)  
**Period**: May 2005 - December 2019  
**Last Updated**: 2024
