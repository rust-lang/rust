#!/bin/sh

while true
do
  df
  memory_pressure
  vm_stat
  sysctl vm.swapusage
  sleep 5
done