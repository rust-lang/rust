# `riscv32{e,em,emc}-unknown-none-elf`

**Tier: 3**

Bare-metal target for RISC-V CPUs with the RV32E, RV32EM and RV32EMC ISAs.

## Target maintainers

[@hegza](https://github.com/hegza)

## Requirements

The target is cross-compiled, and uses static linking. No external toolchain is
required and the default `rust-lld` linker works, but you must specify a linker
script.

## Building the target

This target is included in Rust and can be installed via `rustup`.

## Testing

This is a cross-compiled `no-std` target, which must be run either in a
simulator or by programming them onto suitable hardware. It is not possible to
run the Rust test-suite on this target.

## Cross-compilation toolchains and C code

This target supports C code. If interlinking with C or C++, you may need to use
`riscv32-unknown-elf-gcc` as a linker instead of `rust-lld`.
