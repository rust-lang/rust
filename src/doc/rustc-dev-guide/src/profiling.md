# Profiling the compiler

This section talks about how to profile the compiler and find out where it spends its time.  

Depending on what you're trying to measure, there are several different approaches:

- If you want to see if a PR improves or regresses compiler performance:
  - The [rustc-perf](https://github.com/rust-lang/rustc-perf) project makes this easy and can be triggered to run on a PR via the `@rustc-perf` bot.
  
- If you want a medium-to-high level overview of where `rustc` is spending its time:
  - The `-Zself-profile` flag and [measureme](https://github.com/rust-lang/measureme) tools offer a query-based approach to profiling.
    See [their docs](https://github.com/rust-lang/measureme/blob/master/summarize/Readme.md) for more information.
    
- If you want function level performance data or even just more details than the above approaches:
  - Consider using a native code profiler such as [perf](profiling/with_perf.html) 
  - or [tracy](https://github.com/nagisa/rust_tracy_client) for a nanosecond-precision, 
    full-featured graphical interface.

- If you want a nice visual representation of the compile times of your crate graph, 
  you can use [cargo's `-Ztimings` flag](https://doc.rust-lang.org/cargo/reference/unstable.html#timings), 
  eg. `cargo -Ztimings build`.
  You can use this flag on the compiler itself with `CARGOFLAGS="-Ztimings" ./x.py build`
  
## Optimizing rustc's self-compile-times with cargo-llvm-lines

Using [cargo-llvm-lines](https://github.com/dtolnay/cargo-llvm-lines) you can count the 
number of lines of LLVM IR across all instantiations of a generic function.
Since most of the time compiling rustc is spent in LLVM, the idea is that by
reducing the amount of code passed to LLVM, compiling rustc gets faster.

Example usage:
```
cargo install cargo-llvm-lines
# On a normal crate you could now run `cargo llvm-lines`, but x.py isn't normal :P

# Do a clean before every run, to not mix in the results from previous runs.
./x.py clean
RUSTFLAGS="--emit=llvm-ir" ./x.py build --stage 0 compiler/rustc

# Single crate, eg. rustc_middle
cargo llvm-lines --files ./build/x86_64-unknown-linux-gnu/stage0-rustc/x86_64-unknown-linux-gnu/debug/deps/rustc_middle* > llvm-lines-middle.txt
# Whole compiler at once
cargo llvm-lines --files ./build/x86_64-unknown-linux-gnu/stage0-rustc/x86_64-unknown-linux-gnu/debug/deps/*.ll > llvm-lines.txt
```

Example output:
```
  Lines            Copies        Function name
  -----            ------        -------------
  11802479 (100%)  52848 (100%)  (TOTAL)
   1663902 (14.1%)   400 (0.8%)  rustc_query_system::query::plumbing::get_query_impl::{{closure}}
    683526 (5.8%)  10579 (20.0%) core::ptr::drop_in_place
    568523 (4.8%)    528 (1.0%)  rustc_query_system::query::plumbing::get_query_impl
    472715 (4.0%)   1134 (2.1%)  hashbrown::raw::RawTable<T>::reserve_rehash
    306782 (2.6%)   1320 (2.5%)  rustc_middle::ty::query::plumbing::<impl rustc_query_system::query::QueryContext for rustc_middle::ty::context::TyCtxt>::start_query::{{closure}}::{{closure}}::{{closure}}
    212800 (1.8%)    514 (1.0%)  rustc_query_system::dep_graph::graph::DepGraph<K>::with_task_impl
    194813 (1.7%)    124 (0.2%)  rustc_query_system::query::plumbing::force_query_impl
    158488 (1.3%)      1 (0.0%)  rustc_middle::ty::query::<impl rustc_middle::ty::context::TyCtxt>::alloc_self_profile_query_strings
    119768 (1.0%)    418 (0.8%)  core::ops::function::FnOnce::call_once
    119644 (1.0%)      1 (0.0%)  rustc_target::spec::load_specific
    104153 (0.9%)      7 (0.0%)  rustc_middle::ty::context::_DERIVE_rustc_serialize_Decodable_D_FOR_TypeckResults::<impl rustc_serialize::serialize::Decodable<__D> for rustc_middle::ty::context::TypeckResults>::decode::{{closure}}
     81173 (0.7%)      1 (0.0%)  rustc_middle::ty::query::stats::query_stats
     80306 (0.7%)   2029 (3.8%)  core::ops::function::FnOnce::call_once{{vtable.shim}}
     78019 (0.7%)   1611 (3.0%)  stacker::grow::{{closure}}
     69720 (0.6%)   3286 (6.2%)  <&T as core::fmt::Debug>::fmt
     56327 (0.5%)    186 (0.4%)  rustc_query_system::query::plumbing::incremental_verify_ich
     49714 (0.4%)     14 (0.0%)  rustc_mir::dataflow::framework::graphviz::BlockFormatter<A>::write_node_label
```

Since this doesn't seem to work with incremental compilation or `x.py check`, 
you will be compiling rustc _a lot_.
I recommend changing a few settings in `config.toml` to make it bearable:
```
[rust]
# A debug build takes _a fourth_ as long on my machine, 
# but compiling more than stage0 rustc becomes unbearably slow.
optimize = false

# We can't use incremental anyway, so we disable it for a little speed boost.
incremental = false
# We won't be running it, so no point in compiling debug checks.
debug = false

# Caution: This changes the output of llvm-lines.
# Using a single codegen unit gives more accurate output, but is slower to compile.
# Changing it to the number of cores on my machine increased the output 
# from 3.5GB to 4.1GB and decreased compile times from 5½ min to 4 min.
codegen-units = 1
#codegen-units = 0 # num_cpus
```

What I'm still not sure about is if inlining in MIR optimizations affect llvm-lines.
The output with `-Zmir-opt-level=0` and `-Zmir-opt-level=1` is the same,
but it feels like that some functions that show up at the top should be to small
to have such a high impact. Inlining should only happens in LLVM though.
