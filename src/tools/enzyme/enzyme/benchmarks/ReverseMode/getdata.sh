#!/bin/bash

for bench in "lstm" "ba" "gmm" "ode-const" "ode" "fft" "ode-real"; do
echo $bench
for d in "Tapenade" "Enzyme" "Adept"; do
	echo $d
	cat $bench/results.txt | grep "$d combined" | sed 's/[A-Z a-z]*\([0-9.]\{1,\}\).*/\1/'
done
done
