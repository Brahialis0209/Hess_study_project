# Particle swarm optimization implementation

This repository contains project based on [HESS](https://github.com/Entroforce/Hess)

## Getting started

To build this project:

```bash
./setup.sh
```
## PSO

The PSO method implementation can be found in "my_analog.cpp"

## Example run

Go to hess-empirical-docking and run:

```bash
./dist/Release/GNU-Linux/hess-empirical-docking -r protein_example.pdb -l ligand_example.pdb --autobox_ligand crystal_example.pdb --depth 340 --number_of_iterations 16 --optimize swarm
```
