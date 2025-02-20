# Profiling with rustc-perf

The [Rust benchmark suite][rustc-perf] provides a comprehensive way of profiling and benchmarking
the Rust compiler. You can find instructions on how to use the suite in its [manual][rustc-perf-readme].

However, using the suite manually can be a bit cumbersome. To make this easier for `rustc` contributors,
the compiler build system (`bootstrap`) also provides built-in integration with the benchmarking suite,
which will download and build the suite for you, build a local compiler toolchain and let you profile it using a simplified command-line interface.

You can use the `./x perf <command> [options]` command to use this integration.

You can use normal bootstrap flags for this command, such as `--stage 1` or `--stage 2`, for example to modify the stage of the created sysroot. It might also be useful to configure `bootstrap.toml` to better support profiling, e.g. set `rust.debuginfo-level = 1` to add source line information to the built compiler.

`x perf` currently supports the following commands:
- `benchmark <id>`: Benchmark the compiler and store the results under the passed `id`.
- `compare <baseline> <modified>`: Compare the benchmark results of two compilers with the two passed `id`s.
- `eprintln`: Just run the compiler and capture its `stderr` output. Note that the compiler normally does not print
  anything to `stderr`, you might want to add some `eprintln!` calls to get any output. 
- `samply`: Profile the compiler using the [samply][samply] sampling profiler.
- `cachegrind`: Use [Cachegrind][cachegrind] to generate a detailed simulated trace of the compiler's execution.

> You can find a more detailed description of the profilers in the [`rustc-perf` manual][rustc-perf-readme-profilers].

You can use the following options for the `x perf` command, which mirror the corresponding options of the
`profile_local` and `bench_local` commands that you can use in the suite:

- `--include`: Select benchmarks which should be profiled/benchmarked.
- `--profiles`: Select profiles (`Check`, `Debug`, `Opt`, `Doc`) which should be profiled/benchmarked.
- `--scenarios`: Select scenarios (`Full`, `IncrFull`, `IncrPatched`, `IncrUnchanged`) which should be profiled/benchmarked.

[samply]: https://github.com/mstange/samply
[cachegrind]: https://www.cs.cmu.edu/afs/cs.cmu.edu/project/cmt-40/Nice/RuleRefinement/bin/valgrind-3.2.0/docs/html/cg-manual.html
[rustc-perf]: https://github.com/rust-lang/rustc-perf
[rustc-perf-readme]: https://github.com/rust-lang/rustc-perf/blob/master/collector/README.md
[rustc-perf-readme-profilers]: https://github.com/rust-lang/rustc-perf/blob/master/collector/README.md#profiling-local-builds
