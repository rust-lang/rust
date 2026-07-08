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

## Register class supported types

| Architecture | Register class | Target feature | Allowed types |
| ------------ | -------------- | -------------- | ------------- |
| x86 | `xmm_reg` | `sse` | `i128` |
| x86 | `ymm_reg` | `avx` | `i128` |
| x86 | `zmm_reg` | `avx512f` | `i128` |
| LoongArch | `vreg` | `lsx` | `f32`, `f64`, <br> `i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4`, `f64x2` |
| LoongArch | `xreg` | `lasx` | `f32`, `f64`, <br> `i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4`, `f64x2`, <br> `i8x32`, `i16x16`, `i32x8`, `i64x4`, `f32x8`, `f64x4` |

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
