# Benchmarking rustc

This one is a easier compared to the others. 
All you’re doing is running benchmarks of the compiler itself 
so it’ll build it and run the one set of benchmarks available to it. 
The command is:

   `./x.py bench`

Benchmarking lacks `--no-fail-fast` flag that `test` command has.
   