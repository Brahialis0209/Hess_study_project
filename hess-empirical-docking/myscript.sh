#!/bin/bash

for i in {1..100}
do 
	eval "echo \"iteration ${i}, file PUR2/act_${i}p.pdb\" >> swarm3_result.txt  "
	eval "dist/Release/GNU-Linux/hess-empirical-docking -r PUR2/receptor_p.pdb -l PUR2/act_${i}p.pdb --autobox_ligand PUR2/crystal.pdb --depth 340 --number_of_iterations 8 --optimize swarm | tail -n 3 >> swarm3_result.txt"
	echo "on ${i} iteration, file PUR2/act_${i}p.pdb"

done
