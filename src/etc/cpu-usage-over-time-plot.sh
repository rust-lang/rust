#!/bin/bash

# A small script to help visualizing CPU usage over time data collected on CI
# using `gnuplot`.
#
# This script is expected to be called with two arguments. The first is the full
# commit SHA of the build you're interested in, and the second is the name of
# the builder. For example:
#
#  ./src/etc/cpu-usage-over-time-plot.sh e699ea096fcc2fc9ce8e8bcf884e11496a31cc9f i686-mingw-1
#
# That will generate `$builder.png` in the current directory which you can open
# up to see a hopefully pretty graph.
#
# Improvements to this script are greatly appreciated!

set -ex

bucket=rust-lang-ci2
commit=$1
builder=$2

curl -O https://$bucket.s3.amazonaws.com/rustc-builds/$commit/cpu-$builder.csv

gnuplot <<-EOF
reset
set timefmt '%Y-%m-%dT%H:%M:%S'
set xdata time
set ylabel "CPU Usage %"
set xlabel "Time"
set datafile sep ','
set term png size 3000,1000
set output "$builder.png"
set grid

f(x) = mean_y
fit f(x) 'cpu-$builder.csv' using 1:(100-\$2) via mean_y

set label 1 gprintf("Average = %g%%", mean_y) center font ",18"
set label 1 at graph 0.50, 0.25
set xtics rotate by 45 offset -2,-2.4 300
set ytics 10
set boxwidth 0.5

plot \\
   mean_y with lines linetype 1 linecolor rgb "#ff0000" title "average", \\
   "cpu-$builder.csv" using 1:(100-\$2) with points pointtype 7 pointsize 0.4 title "$builder", \\
   "" using 1:(100-\$2) smooth bezier linewidth 3 title "bezier"
EOF
