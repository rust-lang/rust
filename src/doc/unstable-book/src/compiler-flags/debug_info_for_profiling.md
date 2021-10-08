# `debug-info-for-profiling

---

## Introduction

Automatic Feedback Directed Optimization (AFDO) is a method for using sampling
based profiles to guide optimizations. This is contrasted with other methods of
FDO or profile-guided optimization (PGO) which use instrumented profiling.

Unlike PGO (controlled by the `rustc` flags `-Cprofile-generate` and
`-Cprofile-use`), a binary being profiled does not perform significantly worse,
and thus it's possible to profile binaries used in real workflows and not
necessary to construct artificial workflows.

## Use

In order to use AFDO, the target platform must be Linux running on an `x86_64`
architecture with the performance profiler `perf` available. In addition, the
external tool `create_llvm_prof` from [this repository] must be used.

Given a Rust file `main.rs`, we can produce an optimized binary as follows:

```shell
rustc -O -Zdebug-info-for-profiling main.rs -o main
perf record -b ./main
create_llvm_prof --binary=main --out=code.prof
rustc -O -Zprofile-sample-use=code.prof main.rs -o main2
```

The `perf` command produces a profile `perf.data`, which is then used by the
`create_llvm_prof` command to create `code.prof`. This final profile is then
used by `rustc` to guide optimizations in producing the binary `main2`.

[this repository]: https://github.com/google/autofdo
