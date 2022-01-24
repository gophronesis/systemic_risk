# systemic_risk

Estimate systematic liquidity in the market using only close and volume data:
https://www.financialresearch.gov/working-papers/files/OFRwp-2015-11_Systemwide-Commonalities-in-Market-Liquidity.pdf


## Getting Started

### Dependencies

* See [requirement.txt](requirement.txt)

### Installation

```
pip install systemic_risk
```
or
```
pip install -U systemic_risk
```

### Example

```
import yfinance as yf
from systemic_risk import Liquidity as lq
yf_df = yf.download('SPY, ^FTSE, ^N225', start='2003-01-01', end='2022-01-01')
close, volume = yf_df["Close"].to_numpy(), yf_df["Volume"].to_numpy()
obj = lq.Liquidity(close, volume)
obj.fit_transform()
```


## Authors

Flynn Chen (flynn dot chen AT yale dot edu)


## License

This project is licensed under the MIT License - see the LICENSE file for details
