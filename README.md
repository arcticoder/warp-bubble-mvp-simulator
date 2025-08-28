# Warp Bubble MVP Digital Twin Simulator

Digital-twin simulation suite for warp bubble spacecraft development.

## Features

- **Hardware Digital Twins**: Power, flight computer, sensors, exotic generators
- **Adaptive Fidelity Simulation**: Progressive resolution enhancement from coarse to ultra-fine
- **Monte Carlo Reliability Analysis**: Statistical mission success assessment
- **Real-time Performance Monitoring**: >10 Hz control loops with <1% overhead
- **Pure Software Validation**: 100% simulation-based development without hardware

## Quick Start

```bash
# MVP simulation
python src/simulate_full_warp_MVP.py

# Adaptive fidelity progression
python src/fidelity_runner.py

# Hardware validation demo
python demos/demo_full_warp_simulated_hardware.py
```

## Repository Structure

- `src/`: Core MVP simulation modules
- `tests/`: Validation and integration tests
- `demos/`: Demonstration scripts
- `docs/`: Documentation and specifications
- `examples/`: Usage examples and tutorials
- `config/`: Configuration files and parameters

## Requirements

- Python 3.8+
- NumPy, SciPy
- JAX (optional, for acceleration)
- Core warp-bubble-optimizer framework

## Documentation

See `docs/` for documentation including:
- MVP architecture overview
- Digital twin specifications
- Performance analysis
- Usage examples


## Scope, Validation & Limitations

- Scope: The materials and numeric outputs in this repository are research-stage examples and depend on implementation choices, parameter settings, and numerical tolerances.
- Validation: Reproducibility artifacts (scripts, raw outputs, seeds, and environment details) are provided in `docs/` or `examples/` where available; reproduce analyses with parameter sweeps and independent environments to assess robustness.
- Limitations: Results are sensitive to modeling choices and discretization. Independent verification, sensitivity analyses, and peer review are recommended before using these results for engineering or policy decisions.
