# `riscv32im-unknown-openvm-elf`

**Tier: 3**

OpenVM's zero-knowledge Virtual Machine (zkVM) implementing the RV32IM instruction set.

## Target maintainers

[@jonathanpwang](https://github.com/jonathanpwang)
[@yi-sun](https://github.com/yi-sun)

## Background

This target is an execution environment to produce a verifiable proof of execution of
an RV32IM RISC‑V ELF binary and any output that the developer wishes to make public.
The VM executes the ELF and emits artifacts that can be verified to ensure the integrity
of the computation and its results. This target is implemented in software only; there is
no hardware implementation. See the [OpenVM Book] for architecture and usage.

## Requirements

The target only supports cross-compilation (no host tools). It supports `alloc` with a
default allocator; availability of `std` depends on the OpenVM environment and is not
guaranteed. Binaries are expected to be ELF.

The execution environment is single-threaded, non-preemptive, and does not support
privileged instructions. The binaries expect no operating system and can be thought
of as running on bare-metal. The target does not use `#[target_feature(...)]` or
`-C target-feature=` values. Calling `extern "C"` uses the C calling convention outlined
in the [RISC-V specification].

## Building for the zkVM

You can add this target to the `target` list in `bootstrap.toml` to build with the compiler.
However, most users should follow the tooling and starter templates in the [OpenVM Book],
which provide a streamlined developer experience for building, linking, and packaging
OpenVM guests.

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, either build Rust with the target enabled (see "Building for the zkVM" above)
or use the toolchain and instructions provided in the [OpenVM Book].

## Testing

Note: this target is implemented as a software zkVM; there is no hardware implementation.

The most practical way to test programs is to use the OpenVM host runner and starter
templates described in the [OpenVM Book]. Typically, a "host" program runs on your
development machine (Linux or macOS) and invokes the OpenVM to execute the "guest"
binary compiled for this target, retrieving any public output.

The target currently does not support running the Rust test suite.

## Cross-compilation toolchains and C code

Compatible C code can be built for this target on any compiler that has a RV32IM
target.  On clang and ld.lld linker, it can be generated using the
`-march=rv32im`, `-mabi=ilp32` with llvm features flag `features=+m` and llvm
target `riscv32-unknown-none`.

[RISC-V specification]: https://riscv.org/wp-content/uploads/2015/01/riscv-calling.pdf
[OpenVM]: https://openvm.dev/
[OpenVM Book]: https://docs.openvm.dev/book/getting-started/introduction
