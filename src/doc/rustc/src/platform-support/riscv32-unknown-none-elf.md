# `riscv32{i,im,ima,imc,imac,imafc}-unknown-none-elf`

**Tier: 2**

Bare-metal target for RISC-V CPUs with the RV32I, RV32IM, RV32IMC, RV32IMAFC and RV32IMAC ISAs.

**Tier: 3**

Bare-metal target for RISC-V CPUs with the RV32IMA ISA.

## Target maintainers

* Rust Embedded Working Group, [RISC-V team](https://github.com/rust-embedded/wg#the-risc-v-team)

## Requirements

The target is cross-compiled, and uses static linking. No external toolchain
is required and the default `rust-lld` linker works, but you must specify
a linker script. The [`riscv-rt`] crate provides a suitable one. The
[`riscv-rust-quickstart`] repository gives an example of an RV32 project.

[`riscv-rt`]: https://crates.io/crates/riscv-rt
[`riscv-rust-quickstart`]: https://github.com/riscv-rust/riscv-rust-quickstart

## Building the target

This target is included in Rust and can be installed via `rustup`.

## Testing

This is a cross-compiled `no-std` target, which must be run either in a simulator
or by programming them onto suitable hardware. It is not possible to run the
Rust test-suite on this target.

## Cross-compilation toolchains and C code

This target supports C code. If interlinking with C or C++, you may need to use
`riscv32-unknown-elf-gcc` as a linker instead of `rust-lld`.
