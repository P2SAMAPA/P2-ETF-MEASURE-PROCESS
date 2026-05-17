# Measure-Valued Process Engine

Implements a Fleming‑Viot superprocess approximated by a particle system (100 particles). The distribution of ETF returns evolves stochastically via mutation (random walk) and selection (resampling favouring high returns). The score is the mean of the final particle distribution – a measure of expected return.

- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Particles:** 100
- **Mutation std:** 0.01
- **Selection intensity:** 1.0
- **Output:** top 3 ETFs per universe by predicted return

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
