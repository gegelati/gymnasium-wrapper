# Gymnasium Wrapper

The purpose of this wrapper is to link the TPG [Gegelati library](https://github.com/gegelati/gegelati) to the [Gymnasium](https://gymnasium.farama.org/) reinforcement learning environment.

## How to install

Currently tested only on Mac OS.

### Prerequisites

- An installation of Gegelati: [installation instructions](https://github.com/gegelati/gegelati#hammer_and_wrench-build-and-install-the-library)
- An installation of Gymnasium: [installation instructions](https://github.com/Farama-Foundation/Gymnasium)

### Installation

Clone the repository and navigate in it:
```
git clone https://github.com/gegelati/gynmasium-wrapper.git
cd gynmasium-wrapper
```

Build the example pendulum learning environment:
```
cd bin
cmake ..
cmake --build .
```

Run the example pendulum learning environment:
```
./Release/pendulum
```

Once satisfied, interface your own learning environment from Gymnasium for more adventures!
