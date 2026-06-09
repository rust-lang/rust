# `asm_experimental_arch`

The tracking issue for this feature is: [#133416]

[#133416]: https://github.com/rust-lang/rust/issues/133416

------------------------

This tracks support for additional registers in architectures where inline assembly is already stable.

## Register classes

| Architecture | Register class | Registers | LLVM constraint code |
| ------------ | -------------- | --------- | -------------------- |

## Register class supported types

| Architecture | Register class | Target feature | Allowed types |
| ------------ | -------------- | -------------- | ------------- |
| x86 | `xmm_reg` | `sse` | `i128` |
| x86 | `ymm_reg` | `avx` | `i128` |
| x86 | `zmm_reg` | `avx512f` | `i128` |

## Register aliases

| Architecture | Base register | Aliases |
| ------------ | ------------- | ------- |

## Unsupported registers

| Architecture | Unsupported register | Reason |
| ------------ | -------------------- | ------ |

## Template modifiers

| Architecture | Register class | Modifier | Example output | LLVM modifier |
| ------------ | -------------- | -------- | -------------- | ------------- |
