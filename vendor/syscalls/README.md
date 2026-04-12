# syscalls

[![Crates.io](https://img.shields.io/crates/v/syscalls?style=for-the-badge)](https://crates.io/crates/syscalls)
[![docs.rs](https://img.shields.io/docsrs/syscalls?style=for-the-badge)](https://docs.rs/syscalls)
![License](https://img.shields.io/crates/l/syscalls.svg?style=for-the-badge)

This is a low-level library for listing and invoking raw Linux system calls.

## Features

 - Provides a syscall enum for multiple architectures (see table below).
 - Provides inlinable syscall functions for multiple architectures (see table below).
 - Provides an `Errno` type for Rustic error handling.
 - Provides O(1) array-backed `SysnoSet` and `SysnoMap` types.

## Feature Flags

The features that are enabled by default include `std` and `serde`.

### `std`

By default, `std` support is enabled. If you wish to compile in a `no_std`
environment, use:
```
syscalls = { version = "0.6", default-features = false }
```

### `serde`

Various types can be serialized with Serde. This can be enabled with:
```
syscalls = { version = "0.6", features = ["serde"] }
```

### `full`

Enables all extra features.

### `all`

Enables syscall tables for all architectures. If you don't need all
architectures, you can enable them individually with features like `arm`, `x86`,
`powerpc`, etc. See the Architecture Support table below for a full list of
available architectures.

## Architecture Support

The *Enum* column means that a `Sysno` enum is implemented for this
architecture.

The *Invoke* column means that syscalls can be invoked for this architecture.

The *Stable Rust?* column means that syscall invocation only requires stable
Rust. Some architectures require nightly Rust because inline assembly [is not
yet stabilized for all architectures][asm_experimental_arch].

[asm_experimental_arch]: https://github.com/rust-lang/rust/issues/93335

|    Arch     | Enum  | Invoke  | Stable Rust?  |
|:-----------:|:-----:|:-------:|:-------------:|
|    `arm`\*  |  ✅   |   ✅    |    Yes ✅     |
|  `aarch64`  |  ✅   |   ✅    |    Yes ✅     |
|   `mips`    |  ✅   |   ✅    |     No ❌     |
|  `mips64`   |  ✅   |   ✅    |     No ❌     |
|  `powerpc`  |  ✅   |   ✅    |     No ❌     |
| `powerpc64` |  ✅   |   ✅    |     No ❌     |
|  `riscv32`  |  ✅   |   ❌†   |     No ❌     |
|  `riscv64`  |  ✅   |   ✅    |    Yes ✅     |
|   `s390x`   |  ✅   |   ✅    |     No ❌     |
|   `sparc`   |  ✅   |   ❌    |     N/A       |
|  `sparc64`  |  ✅   |   ❌    |     N/A       |
|    `x86`    |  ✅   |   ✅    |    Yes ✅     |
|  `x86_64`   |  ✅   |   ✅    |    Yes ✅     |

\* Includes ARM thumb mode support.

† Rust does not support riscv32 Linux targets, but syscall functions are
implemented if you're feeling adventurous.

## Updating the syscall list

Updates are pulled from the `.tbl` files in the Linux source tree.

 1. Change the Linux version in `syscalls-gen/src/main.rs` to the latest
    version. Only update to the latest stable version (not release candidates).
 2. Run `cd syscalls-gen && cargo run`. This will regenerate the syscall tables
    in `src/arch/`.
