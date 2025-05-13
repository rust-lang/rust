# `nvptx64-nvidia-cuda`

**Tier: 2**

This is the target meant for deploying code for Nvidia® accelerators based on their CUDA
platform.

## Target maintainers

[@RDambrosio016](https://github.com/RDambrosio016)
[@kjetilkjeka](https://github.com/kjetilkjeka)

## Requirements

This target is `no_std` and will typically be built with crate-type `cdylib` and `-C linker-flavor=llbc`, which generates PTX.
The necessary components for this workflow are:

- `rustup toolchain add nightly`
- `rustup component add llvm-tools --toolchain nightly`
- `rustup component add llvm-bitcode-linker --toolchain nightly`

There are two options for using the core library:

- `rustup component add rust-src --toolchain nightly` and build using `-Z build-std=core`.
- `rustup target add nvptx64-nvidia-cuda --toolchain nightly`

### Target and features

It is necessary to specify the target, such as `-C target-cpu=sm_89`. This implies two target features: `sm_89` and `ptx78` (and all preceding features within `sm_*` and `ptx*`). Rust will default to using the oldest PTX version that supports the target processor (see [this table](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes-ptx-release-history)), which maximizes driver compatibility.
One can use `-C target-feature=+ptx80` to choose a later PTX version without changing the target (the default `ptx78` requires CUDA driver version 11.8, while `ptx80` would require driver version 12.0).

Although `ptx*` is represented as a target feature, it is a compile-time property and it is not possible to build a crate that uses instructions not present in the PTX version specified at compile-time (either via `target-cpu` or `target-feature`).
For example, consider an unaligned barrier `barrier.sync`, which requires both `sm_70` and `ptx60`.
If one wants to support building for older devices (e.g., `-C target-cpu=sm_62`; ensuring that this unaligned barrier is unreachable at run-time on such devices), the relevant function could use attributes:
```
#[cfg(target_feature = "ptx60")]
#[target_feature(enable = "sm_70")]
```

## Building Rust kernels

A `no_std` crate containing one or more functions with `extern "ptx-kernel"` can be compiled to PTX using a command like the following.

```console
$ cargo +nightly rustc --target=nvptx64-nvidia-cuda -Zbuild-std=core --crate-type=cdylib -- -Clinker-flavor=llbc -Ctarget-cpu=sm_89 -Zunstable-options
```
<!-- FIXME: fill this out

## Requirements

Does the target support host tools, or only cross-compilation? Does the target
support std, or alloc (either with a default allocator, or if the user supplies
an allocator)?

Document the expectations of binaries built for the target. Do they assume
specific minimum features beyond the baseline of the CPU/environment/etc? What
version of the OS or environment do they expect?

Are there notable `#[target_feature(...)]` or `-C target-feature=` values that
programs may wish to use?

What calling convention does `extern "C"` use on the target?

What format do binaries use by default? ELF, PE, something else?

## Building the target

If Rust doesn't build the target by default, how can users build it? Can users
just add it to the `target` list in `bootstrap.toml`?

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing

Does the target support running binaries, or do binaries have varying
expectations that prevent having a standard way to run them? If users can run
binaries, can they do so in some common emulator, or do they need native
hardware? Does the target support running the Rust testsuite?

## Cross-compilation toolchains and C code

Does the target support C code? If so, what toolchain target should users use
to build compatible C code? (This may match the target triple, or it may be a
toolchain for a different target triple, potentially with specific options or
caveats.)

-->
