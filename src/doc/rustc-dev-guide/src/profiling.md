# Profiling the compiler

This section talks about how to profile the compiler and find out where it spends its time.

Depending on what you're trying to measure, there are several different approaches:

- If you want to see if a PR improves or regresses compiler performance,
  see the [rustc-perf chapter](tests/perf.md) for requesting a benchmarking run.

- If you want a medium-to-high level overview of where `rustc` is spending its time:
  - The `-Z self-profile` flag and [measureme](https://github.com/rust-lang/measureme) tools offer a query-based approach to profiling.
    See [their docs](https://github.com/rust-lang/measureme/blob/master/summarize/README.md) for more information.

- If you want function level performance data or even just more details than the above approaches:
  - Consider using a native code profiler such as [perf](profiling/with_perf.md)
  - or [tracy](https://github.com/nagisa/rust_tracy_client) for a nanosecond-precision,
    full-featured graphical interface.

- If you want a nice visual representation of the compile times of your crate graph,
  you can use [cargo's `--timings` flag](https://doc.rust-lang.org/nightly/cargo/reference/timings.html),
  e.g. `cargo build --timings`.
  You can use this flag on the compiler itself with `CARGOFLAGS="--timings" ./x build`

- If you want to profile memory usage, you can use various tools depending on what operating system
  you are using.
  - For Windows, read our [WPA guide](profiling/wpa_profiling.md).

## Optimizing rustc's bootstrap times with `cargo-llvm-lines`

Using [cargo-llvm-lines](https://github.com/dtolnay/cargo-llvm-lines) you can count the
number of lines of LLVM IR across all instantiations of a generic function.
Since most of the time compiling rustc is spent in LLVM, the idea is that by
reducing the amount of code passed to LLVM, compiling rustc gets faster.

To use `cargo-llvm-lines` together with somewhat custom rustc build process, you can use
`-C save-temps` to obtain required LLVM IR. The option preserves temporary work products
created during compilation. Among those is LLVM IR that represents an input to the
optimization pipeline; ideal for our purposes. It is stored in files with `*.no-opt.bc`
extension in LLVM bitcode format.

Example usage:
```
cargo install cargo-llvm-lines
# On a normal crate you could now run `cargo llvm-lines`, but `x` isn't normal :P

# Do a clean before every run, to not mix in the results from previous runs.
./x clean
env RUSTFLAGS=-Csave-temps ./x build --stage 0 compiler/rustc

# Single crate, e.g., rustc_middle. (Relies on the glob support of your shell.)
# Convert unoptimized LLVM bitcode into a human readable LLVM assembly accepted by cargo-llvm-lines.
for f in build/x86_64-unknown-linux-gnu/stage0-rustc/x86_64-unknown-linux-gnu/release/deps/rustc_middle-*.no-opt.bc; do
  ./build/x86_64-unknown-linux-gnu/llvm/bin/llvm-dis "$f"
done
cargo llvm-lines --files ./build/x86_64-unknown-linux-gnu/stage0-rustc/x86_64-unknown-linux-gnu/release/deps/rustc_middle-*.ll > llvm-lines-middle.txt

# Specify all crates of the compiler.
for f in build/x86_64-unknown-linux-gnu/stage0-rustc/x86_64-unknown-linux-gnu/release/deps/*.no-opt.bc; do
  ./build/x86_64-unknown-linux-gnu/llvm/bin/llvm-dis "$f"
done
cargo llvm-lines --files ./build/x86_64-unknown-linux-gnu/stage0-rustc/x86_64-unknown-linux-gnu/release/deps/*.ll > llvm-lines.txt
```

Example output for the compiler:
```
  Lines            Copies          Function name
  -----            ------          -------------
  45207720 (100%)  1583774 (100%)  (TOTAL)
   2102350 (4.7%)   146650 (9.3%)  core::ptr::drop_in_place
    615080 (1.4%)     8392 (0.5%)  std::thread::local::LocalKey<T>::try_with
    594296 (1.3%)     1780 (0.1%)  hashbrown::raw::RawTable<T>::rehash_in_place
    592071 (1.3%)     9691 (0.6%)  core::option::Option<T>::map
    528172 (1.2%)     5741 (0.4%)  core::alloc::layout::Layout::array
    466854 (1.0%)     8863 (0.6%)  core::ptr::swap_nonoverlapping_one
    412736 (0.9%)     1780 (0.1%)  hashbrown::raw::RawTable<T>::resize
    367776 (0.8%)     2554 (0.2%)  alloc::raw_vec::RawVec<T,A>::grow_amortized
    367507 (0.8%)      643 (0.0%)  rustc_query_system::dep_graph::graph::DepGraph<K>::with_task_impl
    355882 (0.8%)     6332 (0.4%)  alloc::alloc::box_free
    354556 (0.8%)    14213 (0.9%)  core::ptr::write
    354361 (0.8%)     3590 (0.2%)  core::iter::traits::iterator::Iterator::fold
    347761 (0.8%)     3873 (0.2%)  rustc_middle::ty::context::tls::set_tlv
    337534 (0.7%)     2377 (0.2%)  alloc::raw_vec::RawVec<T,A>::allocate_in
    331690 (0.7%)     3192 (0.2%)  hashbrown::raw::RawTable<T>::find
    328756 (0.7%)     3978 (0.3%)  rustc_middle::ty::context::tls::with_context_opt
    326903 (0.7%)      642 (0.0%)  rustc_query_system::query::plumbing::try_execute_query
```

Since this doesn't seem to work with incremental compilation or `./x check`,
you will be compiling rustc _a lot_.
I recommend changing a few settings in `bootstrap.toml` to make it bearable:
```
[rust]
# A debug build takes _a third_ as long on my machine,
# but compiling more than stage0 rustc becomes unbearably slow.
optimize = false

# We can't use incremental anyway, so we disable it for a little speed boost.
incremental = false
# We won't be running it, so no point in compiling debug checks.
debug = false

# Using a single codegen unit gives less output, but is slower to compile.
codegen-units = 0  # num_cpus
```

The llvm-lines output is affected by several options.
`optimize = false` increases it from 2.1GB to 3.5GB and `codegen-units = 0` to 4.1GB.

MIR optimizations have little impact. Compared to the default `RUSTFLAGS="-Z
mir-opt-level=1"`, level 0 adds 0.3GB and level 2 removes 0.2GB.
As of <!-- date-check --> July 2022,
inlining happens in LLVM and GCC codegen backends,
missing only in the Cranelift one.
