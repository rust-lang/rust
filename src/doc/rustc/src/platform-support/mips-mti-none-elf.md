# `mips*-mti-none-elf`

**Tier: 3**

MIPS32r2 baremetal softfloat, Big Endian or Little Endian.

- mips-mti-none-elf
- mipsel-mti-none-elf

## Target maintainers

[@wzssyqa](https://github.com/wzssyqa)

## Background

These 2 targets, aka mips-mti-none-elf and mipsel-mti-none-elf, are for
baremetal development of MIPS32r2. The lld is used instead of Gnu-ld.

## Requirements

The target only supports cross compilation and no host tools. The target
supports `alloc` with a default allocator while only support `no-std` development.

The vendor name `mti` follows the naming of gcc to indicate MIPS32r2.

## Cross-compilation toolchains and C code

Compatible C code can be built for this target on any compiler that has a MIPS32r2
target.  On clang and ld.lld linker, it can be generated using the
`-march=mips`/`-march=mipsel`, `-mabi=32` with llvm features flag
`features=+mips32r2,+soft-float,+noabicalls`.
