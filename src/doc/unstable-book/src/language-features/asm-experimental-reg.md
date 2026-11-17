# `asm_experimental_reg`

The tracking issue for this feature is: [#133416]

[#133416]: https://github.com/rust-lang/rust/issues/133416

------------------------

This tracks support for additional registers in architectures where inline assembly is already stable.

## Register classes

| Architecture | Register class | Registers | LLVM constraint code |
| ------------ | -------------- | --------- | -------------------- |
| LoongArch | `vreg` | `$vr[0-31]` | `f` |
| LoongArch | `xreg` | `$xr[0-31]` | `f` |
| AArch64 | `preg` | `p[0-16]` | `Upa` |

## Register class supported types

| Architecture | Register class | Target feature | Allowed types |
| ------------ | -------------- | -------------- | ------------- |
| LoongArch | `vreg` | `lsx` | `i128`, `f32`, `f64`, <br> `i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4`, `f64x2` |
| RISC-V | `vreg` | `v` | `i8x16`, `i16x8`, `i32x4`, `i64x2`, `i8x32`, `i16x16`, `i32x8`, `i64x4`, `i8x128`, `i16x64`, `i32x32`, `i64x16` |
| LoongArch | `xreg` | `lasx` | `i128`, `f32`, `f64`, <br> `i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4`, `f64x2`, <br> `i8x32`, `i16x16`, `i32x8`, `i64x4`, `f32x8`, `f64x4` |
| AArch64 | `vreg` | `sve` | `i8xN`, `i16xB`, `i32xN`, `i64xN`, `f16xN`, `f32xN`, `f64xN` (scalable vector) |
| AArch64 | `preg` | `sve` | `i1xN` (scalable vector predicate) |

## Register aliases

| Architecture | Base register | Aliases |
| ------------ | ------------- | ------- |
| LoongArch | `$f[0-7]` | `$fa[0-7]`, `$vr[0-7]`, `$xr[0-7]` |
| LoongArch | `$f[8-23]` | `$ft[0-15]`, `$vr[8-23]`, `$xr[8-23]` |
| LoongArch | `$f[24-31]` | `$fs[0-7]`, `$vr[24-31]`, `$xr[24-31]` |

## Unsupported registers

| Architecture | Unsupported register | Reason |
| ------------ | -------------------- | ------ |

## Template modifiers

| Architecture | Register class | Modifier | Example output | LLVM modifier |
| ------------ | -------------- | -------- | -------------- | ------------- |
| LoongArch | `freg` | `w` | `$vr0` | `w` |
| LoongArch | `freg` | `u` | `$xr0` | `u` |
| LoongArch | `vreg` | None | `$vr0` | `w` |
| LoongArch | `vreg` | `u` | `$xr0` | `u` |
| LoongArch | `xreg` | None | `$xr0` | `u` |
| LoongArch | `xreg` | `w` | `$vr0` | `w` |
| AArch64 | `vreg` | `z` | `z0` | `z` |

### RISC-V vector registers

`vreg` operands require the `v` target feature and the `asm_experimental_reg`
gate. Allowed types are fixed-size SIMD vectors up to 1024 bytes; LLVM chooses
the LMUL (including fractional groups) from the vector length and the target's
minimum VLEN. LLVM's multi-letter constraint needs the `^` prefix (`^vr`); plain
`vr` is parsed as two constraints and fails register allocation.

RISC-V vector asm is emitted with `~{vtype}` and `~{vl}` (plus `~{vxsat}`,
`~{vxrm}`) clobbers, so LLVM does not assume a `vsetvli` from one block survives
the next. Codegen may also spill `vreg` operands between blocks and emit its own
`vsetvli` for the spill. Set the required SEW/LMUL with `vsetvli` in every vector
asm block that depends on it; do not rely on a single `vsetvli` earlier in the
function.
