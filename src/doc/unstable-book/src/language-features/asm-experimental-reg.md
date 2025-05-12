# `asm_experimental_arch`

The tracking issue for this feature is: [#133416]

[#133416]: https://github.com/rust-lang/rust/issues/133416

------------------------

This tracks support for additional registers in architectures where inline assembly is already stable.

## Register classes

| Architecture | Register class | Registers | LLVM constraint code |
| ------------ | -------------- | --------- | -------------------- |
| s390x | `vreg` | `v[0-31]` | `v` |

> **Notes**:
> - s390x `vreg` is clobber-only in stable.

## Register class supported types

| Architecture | Register class | Target feature | Allowed types |
| ------------ | -------------- | -------------- | ------------- |
| s390x | `vreg` | `vector` | `i32`, `f32`, `i64`, `f64`, `i128`, `f128`, `i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4`, `f64x2` |

## Register aliases

| Architecture | Base register | Aliases |
| ------------ | ------------- | ------- |

## Unsupported registers

| Architecture | Unsupported register | Reason |
| ------------ | -------------------- | ------ |

## Template modifiers

| Architecture | Register class | Modifier | Example output | LLVM modifier |
| ------------ | -------------- | -------- | -------------- | ------------- |
| s390x | `vreg` | None | `%v0` | None |
