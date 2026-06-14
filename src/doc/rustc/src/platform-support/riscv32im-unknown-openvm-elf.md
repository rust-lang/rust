# `riscv32im-unknown-openvm-elf`

**Tier: 3**

Target for [OpenVM](https://github.com/openvm-org/openvm/) virtual machines with the RV32IM ISA and custom RISC-V extensions defined through OpenVM's extension framework.

## Target maintainers

[@jonathanpwang](https://github.com/jonathanpwang)
[@yi-sun](https://github.com/yi-sun)

## Background

This target is an execution environment to produce a verifiable cryptographic proof of execution of
a RISC‑V ELF binary and any output that the developer wishes to make public.
The execution environment is implemented as a virtual machine in software only. The target is not intended for bare metal hardware. The virtual machine may be extended to support custom RISC-V instruction sets, which may be invoked from Rust via the `asm!` macro. See the [OpenVM Book] for further documentation on the architecture and usage.

We provide a cargo extension called [cargo-openvm] that provides tools for cross-compilation, execution, and generating cryptographic proofs of execution.

## Requirements

The target supports cross-compilation from any host and does not support host tools. It supports `alloc` with a
default allocator. Partial support for the Rust `std` library is provided using custom RISC-V instructions and requires the `openvm` crate with the `"std"` feature enabled. Further details and limitations of `std` support are documented [here](https://docs.openvm.dev/book/writing-apps/writing-a-program#rust-std-library-support).

The target's execution environment is single-threaded, non-preemptive, and does not support
privileged instructions. At present, unaligned accesses are not supported and will result in execution traps. The binaries expect no operating system and can be thought
of as running on bare-metal. The target does not use `#[target_feature(...)]` or
`-C target-feature=` values.

Binaries are expected to be ELF.

Calling `extern "C"` uses the C calling convention outlined
in the [RISC-V specification].

## Building the target
The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["riscv32im-unknown-openvm-elf"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will need to do one of the following:
- Build Rust with the target enabled (see "Building the target" above)
- Build your own copy of `core` by passing args `-Zbuild-std=alloc,core,proc_macro,panic_abort,std -Zbuild-std-features=compiler-builtins-mem`
- Use `cargo openvm build` provided by the cargo extension [cargo-openvm].

The `cargo-openvm` utility is a command-line interface that calls `cargo` with the `build-std` flags above together with `rustc` flags `-C passes=lower-atomic -C link-arg=-Ttext=<TEXT_START>` to map text to the appropriate location. The text start (presently `0x0020_0800`) must be set to start above the stack top, and heap begins right after the text. The utility also includes the `rustc` flag `--cfg getrandom_backend="custom"` to enable a custom backend for the `getrandom` crate.

## Testing

Note: this target is implemented as a software virtual machine; there is no hardware implementation.

Guest programs cross-compiled to the target must be run on the host inside OpenVM virtual machines, which are software emulators. The most practical way to do this is via either the [cargo-openvm] command-line interface or the [OpenVM SDK].

The target currently does not support running the Rust test suite.

## Cross-compilation toolchains and C code

Compatible C code can be built for this target on any compiler that has a RV32IM
target.  On clang and ld.lld linker, it can be generated using the
`-march=rv32im`, `-mabi=ilp32` with llvm features flag `features=+m` and llvm
target `riscv32-unknown-none`.

[RISC-V specification]: https://riscv.org/wp-content/uploads/2015/01/riscv-calling.pdf
[OpenVM]: https://github.com/openvm-org/openvm/
[OpenVM Book]: https://docs.openvm.dev/book/
[OpenVM SDK]: https://docs.openvm.dev/book/advanced-usage/sdk
[cargo-openvm]: https://docs.openvm.dev/book/getting-started/install
