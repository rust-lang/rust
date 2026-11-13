#!/bin/sh

while true
do
  df
  top -l 1 -s 0 | grep PhysMem
  sleep 5
done