# Optimized build pipeline
This binary implements a heavily optimized build pipeline for `rustc` and `LLVM` artifacts that are used for both for
benchmarking using the perf. bot and for final distribution to users.

It uses LTO, PGO and BOLT to optimize the compiler and LLVM as much as possible.
This logic is not part of bootstrap, because it needs to invoke bootstrap multiple times, force-rebuild various
artifacts repeatedly and sometimes go around bootstrap's cache mechanism.
