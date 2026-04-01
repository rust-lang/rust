# proc-macro-srv-cli

A standalone binary for the `proc-macro-srv` crate that provides procedural macro expansion for rust-analyzer.

## Overview

rust-analyzer uses a RPC (via stdio) client-server architecture for procedural macro expansion. This is necessary because:

1. Proc macros are dynamic libraries that can segfault, bringing down the entire process, so running them out of process allows rust-analyzer to recover from fatal errors.
2. Proc macro dylibs are compiled against a specific rustc version and require matching internal APIs to load and execute, as such having this binary shipped as a rustup component allows us to always match the rustc version irrespective of the rust-analyzer version used.

## The `sysroot-abi` Feature

**The `sysroot-abi` feature is required for the binary to actually function.** Without it, the binary will return an error:

```
proc-macro-srv-cli needs to be compiled with the `sysroot-abi` feature to function
```

This feature is necessary because the proc-macro server needs access to unstable rustc internals (`proc_macro_internals`, `proc_macro_diagnostic`, `proc_macro_span`) which are only available on nightly or with `RUSTC_BOOTSTRAP=1`.
rust-analyzer is a stable toolchain project though, so the feature flag is used to have it remain compilable on stable by default.

### Building

```bash
# Using nightly toolchain
cargo build -p proc-macro-srv-cli --features sysroot-abi

# Or with RUSTC_BOOTSTRAP on stable
RUSTC_BOOTSTRAP=1 cargo build -p proc-macro-srv-cli --features sysroot-abi
```

### Installing the proc-macro server

For local testing purposes, you can install the proc-macro server using the xtask command:

```bash
# Recommended: use the xtask command
cargo xtask install --proc-macro-server
```

## Testing

```bash
cargo test --features sysroot-abi -p proc-macro-srv -p proc-macro-srv-cli -p proc-macro-api
```

The tests use a test proc macro dylib built by the `proc-macro-test` crate, which compiles a small proc macro implementation during build time.

**Note**: Tests only compile on nightly toolchains (or with `RUSTC_BOOTSTRAP=1`).

## Usage

The binary requires the `RUST_ANALYZER_INTERNALS_DO_NOT_USE` environment variable to be set. This is intentional—the binary is an implementation detail of rust-analyzer and its API is still unstable:

```bash
RUST_ANALYZER_INTERNALS_DO_NOT_USE=1 rust-analyzer-proc-macro-srv --version
```

## Related Crates

- `proc-macro-srv`: The core server library that handles loading dylibs and expanding macros, but not the RPC protocol.
- `proc-macro-api`: The client library used by rust-analyzer to communicate with this server as well as the protocol definitions.
- `proc-macro-test`: Test harness with sample proc macros for testing
- `proc-macro-srv-cli`: The actual server binary that handles the RPC protocol.
