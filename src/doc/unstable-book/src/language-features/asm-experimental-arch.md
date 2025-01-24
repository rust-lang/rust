# `asm_experimental_arch`

The tracking issue for this feature is: [#93335]

[#93335]: https://github.com/rust-lang/rust/issues/93335

------------------------

This feature tracks `asm!` and `global_asm!` support for the following architectures:
- NVPTX
- PowerPC
- Hexagon
- MIPS32r2 and MIPS64r2
- wasm32
- BPF
- SPIR-V
- AVR
- MSP430
- M68k
- CSKY
- SPARC

## Register classes

| Architecture | Register class | Registers                          | LLVM constraint code |
| ------------ | -------------- | ---------------------------------- | -------------------- |
| MIPS         | `reg`          | `$[2-25]`                          | `r`                  |
| MIPS         | `freg`         | `$f[0-31]`                         | `f`                  |
| NVPTX        | `reg16`        | None\*                             | `h`                  |
| NVPTX        | `reg32`        | None\*                             | `r`                  |
| NVPTX        | `reg64`        | None\*                             | `l`                  |
| Hexagon      | `reg`          | `r[0-28]`                          | `r`                  |
| Hexagon      | `preg`         | `p[0-3]`                           | Only clobbers        |
| PowerPC      | `reg`          | `r0`, `r[3-12]`, `r[14-28]`        | `r`                  |
| PowerPC      | `reg_nonzero`  | `r[3-12]`, `r[14-28]`              | `b`                  |
| PowerPC      | `freg`         | `f[0-31]`                          | `f`                  |
| PowerPC      | `vreg`         | `v[0-31]`                          | `v`                  |
| PowerPC      | `cr`           | `cr[0-7]`, `cr`                    | Only clobbers        |
| PowerPC      | `xer`          | `xer`                              | Only clobbers        |
| wasm32       | `local`        | None\*                             | `r`                  |
| BPF          | `reg`          | `r[0-10]`                          | `r`                  |
| BPF          | `wreg`         | `w[0-10]`                          | `w`                  |
| AVR          | `reg`          | `r[2-25]`, `XH`, `XL`, `ZH`, `ZL`  | `r`                  |
| AVR          | `reg_upper`    | `r[16-25]`, `XH`, `XL`, `ZH`, `ZL` | `d`                  |
| AVR          | `reg_pair`     | `r3r2` .. `r25r24`, `X`, `Z`       | `r`                  |
| AVR          | `reg_iw`       | `r25r24`, `X`, `Z`                 | `w`                  |
| AVR          | `reg_ptr`      | `X`, `Z`                           | `e`                  |
| MSP430       | `reg`          | `r[0-15]`                          | `r`                  |
| M68k         | `reg`          | `d[0-7]`, `a[0-7]`                 | `r`                  |
| M68k         | `reg_data`     | `d[0-7]`                           | `d`                  |
| M68k         | `reg_addr`     | `a[0-3]`                           | `a`                  |
| CSKY         | `reg`          | `r[0-31]`                          | `r`                  |
| CSKY         | `freg`         | `f[0-31]`                          | `f`                  |
| SPARC        | `reg`          | `r[2-29]`                          | `r`                  |
| SPARC        | `yreg`         | `y`                                | Only clobbers        |

> **Notes**:
> - NVPTX doesn't have a fixed register set, so named registers are not supported.
>
> - WebAssembly doesn't have registers, so named registers are not supported.

# Register class supported types

| Architecture | Register class                  | Target feature | Allowed types                           |
| ------------ | ------------------------------- | -------------- | --------------------------------------- |
| MIPS32       | `reg`                           | None           | `i8`, `i16`, `i32`, `f32`               |
| MIPS32       | `freg`                          | None           | `f32`, `f64`                            |
| MIPS64       | `reg`                           | None           | `i8`, `i16`, `i32`, `i64`, `f32`, `f64` |
| MIPS64       | `freg`                          | None           | `f32`, `f64`                            |
| NVPTX        | `reg16`                         | None           | `i8`, `i16`                             |
| NVPTX        | `reg32`                         | None           | `i8`, `i16`, `i32`, `f32`               |
| NVPTX        | `reg64`                         | None           | `i8`, `i16`, `i32`, `f32`, `i64`, `f64` |
| Hexagon      | `reg`                           | None           | `i8`, `i16`, `i32`, `f32`               |
| Hexagon      | `preg`                          | N/A            | Only clobbers                           |
| PowerPC      | `reg`                           | None           | `i8`, `i16`, `i32`, `i64` (powerpc64 only) |
| PowerPC      | `reg_nonzero`                   | None           | `i8`, `i16`, `i32`, `i64` (powerpc64 only) |
| PowerPC      | `freg`                          | None           | `f32`, `f64`                            |
| PowerPC      | `vreg`                          | `altivec`      | `i8x16`, `i16x8`, `i32x4`, `f32x4`      |
| PowerPC      | `vreg`                          | `vsx`          | `f32`, `f64`, `i64x2`, `f64x2`          |
| PowerPC      | `cr`                            | N/A            | Only clobbers                           |
| PowerPC      | `xer`                           | N/A            | Only clobbers                           |
| wasm32       | `local`                         | None           | `i8` `i16` `i32` `i64` `f32` `f64`      |
| BPF          | `reg`                           | None           | `i8` `i16` `i32` `i64`                  |
| BPF          | `wreg`                          | `alu32`        | `i8` `i16` `i32`                        |
| AVR          | `reg`, `reg_upper`              | None           | `i8`                                    |
| AVR          | `reg_pair`, `reg_iw`, `reg_ptr` | None           | `i16`                                   |
| MSP430       | `reg`                           | None           | `i8`, `i16`                             |
| M68k         | `reg`, `reg_addr`               | None           | `i16`, `i32`                            |
| M68k         | `reg_data`                      | None           | `i8`, `i16`, `i32`                      |
| CSKY         | `reg`                           | None           | `i8`, `i16`, `i32`                      |
| CSKY         | `freg`                          | None           | `f32`,                                  |
| SPARC        | `reg`                           | None           | `i8`, `i16`, `i32`, `i64` (SPARC64 only) |
| SPARC        | `yreg`                          | N/A            | Only clobbers                           |

## Register aliases

| Architecture | Base register | Aliases   |
| ------------ | ------------- | --------- |
| Hexagon      | `r29`         | `sp`      |
| Hexagon      | `r30`         | `fr`      |
| Hexagon      | `r31`         | `lr`      |
| PowerPC      | `r1`          | `sp`      |
| PowerPC      | `r31`         | `fp`      |
| PowerPC      | `r[0-31]`     | `[0-31]`  |
| PowerPC      | `f[0-31]`     | `fr[0-31]`|
| BPF          | `r[0-10]`     | `w[0-10]` |
| AVR          | `XH`          | `r27`     |
| AVR          | `XL`          | `r26`     |
| AVR          | `ZH`          | `r31`     |
| AVR          | `ZL`          | `r30`     |
| MSP430       | `r0`          | `pc`      |
| MSP430       | `r1`          | `sp`      |
| MSP430       | `r2`          | `sr`      |
| MSP430       | `r3`          | `cg`      |
| MSP430       | `r4`          | `fp`      |
| M68k         | `a5`          | `bp`      |
| M68k         | `a6`          | `fp`      |
| M68k         | `a7`          | `sp`, `usp`, `ssp`, `isp` |
| CSKY         | `r[0-3]`      | `a[0-3]`  |
| CSKY         | `r[4-11]`     | `l[0-7]`  |
| CSKY         | `r[12-13]`    | `t[0-1]`  |
| CSKY         | `r14`         | `sp`      |
| CSKY         | `r15`         | `lr`      |
| CSKY         | `r[16-17]`    | `l[8-9]`  |
| CSKY         | `r[18-25]`    | `t[2-9]`  |
| CSKY         | `r28`         | `rgb`     |
| CSKY         | `r29`         | `rtb`     |
| CSKY         | `r30`         | `svbr`    |
| CSKY         | `r31`         | `tls`     |
| SPARC        | `r[0-7]`      | `g[0-7]`  |
| SPARC        | `r[8-15]`     | `o[0-7]`  |
| SPARC        | `r[16-23]`    | `l[0-7]`  |
| SPARC        | `r[24-31]`    | `i[0-7]`  |

> **Notes**:
> - TI does not mandate a frame pointer for MSP430, but toolchains are allowed
    to use one; LLVM uses `r4`.

## Unsupported registers

| Architecture | Unsupported register                    | Reason                                                                                                                                                                              |
| ------------ | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| All          | `sp`, `r14`/`o6` (SPARC)                | The stack pointer must be restored to its original value at the end of an asm code block.                                                                                           |
| All          | `fr` (Hexagon), `fp` (PowerPC), `$fp` (MIPS), `Y` (AVR), `r4` (MSP430), `a6` (M68k), `r30`/`i6` (SPARC) | The frame pointer cannot be used as an input or output.                                                             |
| All          | `r19` (Hexagon), `r29` (PowerPC), `r30` (PowerPC) | These are used internally by LLVM as "base pointer" for functions with complex stack frames.                                                                              |
| MIPS         | `$0` or `$zero`                         | This is a constant zero register which can't be modified.                                                                                                                           |
| MIPS         | `$1` or `$at`                           | Reserved for assembler.                                                                                                                                                             |
| MIPS         | `$26`/`$k0`, `$27`/`$k1`                | OS-reserved registers.                                                                                                                                                              |
| MIPS         | `$28`/`$gp`                             | Global pointer cannot be used as inputs or outputs.                                                                                                                                 |
| MIPS         | `$ra`                                   | Return address cannot be used as inputs or outputs.                                                                                                                                 |
| Hexagon      | `lr`                                    | This is the link register which cannot be used as an input or output.                                                                                                               |
| PowerPC      | `r2`, `r13`                             | These are system reserved registers.                                                                                                                                                |
| PowerPC      | `lr`                                    | The link register cannot be used as an input or output.                                                                                                                             |
| PowerPC      | `ctr`                                   | The counter register cannot be used as an input or output.                                                                                                                          |
| PowerPC      | `vrsave`                                | The vrsave register cannot be used as an input or output.                                                                                                                           |
| AVR          | `r0`, `r1`, `r1r0`                      | Due to an issue in LLVM, the `r0` and `r1` registers cannot be used as inputs or outputs.  If modified, they must be restored to their original values before the end of the block. |
|MSP430        | `r0`, `r2`, `r3`                        | These are the program counter, status register, and constant generator respectively. Neither the status register nor constant generator can be written to.                          |
| M68k         | `a4`, `a5`                              | Used internally by LLVM for the base pointer and global base pointer. |
| CSKY         | `r7`, `r28`                             | Used internally by LLVM for the base pointer and global base pointer. |
| CSKY         | `r8`                                    | Used internally by LLVM for the frame pointer. |
| CSKY         | `r14`                                   | Used internally by LLVM for the stack pointer. |
| CSKY         | `r15`                                   | This is the link register. |
| CSKY         | `r[26-30]`                              | Reserved by its ABI.       |
| CSKY         | `r31`                                   | This is the TLS register.  |
| SPARC        | `r0`/`g0`                               | This is always zero and cannot be used as inputs or outputs. |
| SPARC        | `r1`/`g1`                               | Used internally by LLVM. |
| SPARC        | `r5`/`g5`                               | Reserved for system. (SPARC32 only) |
| SPARC        | `r6`/`g6`, `r7`/`g7`                    | Reserved for system. |
| SPARC        | `r31`/`i7`                              | Return address cannot be used as inputs or outputs. |


## Template modifiers

| Architecture | Register class | Modifier | Example output | LLVM modifier |
| ------------ | -------------- | -------- | -------------- | ------------- |
| MIPS         | `reg`          | None     | `$2`           | None          |
| MIPS         | `freg`         | None     | `$f0`          | None          |
| NVPTX        | `reg16`        | None     | `rs0`          | None          |
| NVPTX        | `reg32`        | None     | `r0`           | None          |
| NVPTX        | `reg64`        | None     | `rd0`          | None          |
| Hexagon      | `reg`          | None     | `r0`           | None          |
| PowerPC      | `reg`          | None     | `0`            | None          |
| PowerPC      | `reg_nonzero`  | None     | `3`            | None          |
| PowerPC      | `freg`         | None     | `0`            | None          |
| PowerPC      | `vreg`         | None     | `0`            | None          |
| SPARC        | `reg`          | None     | `%o0`          | None          |
| CSKY         | `reg`          | None     | `r0`           | None          |
| CSKY         | `freg`         | None     | `f0`           | None          |

# Flags covered by `preserves_flags`

These flags registers must be restored upon exiting the asm block if the `preserves_flags` option is set:
- AVR
  - The status register `SREG`.
- MSP430
  - The status register `r2`.
- M68k
  - The condition code register `ccr`.
- SPARC
  - Integer condition codes (`icc` and `xcc`)
  - Floating-point condition codes (`fcc[0-3]`)
