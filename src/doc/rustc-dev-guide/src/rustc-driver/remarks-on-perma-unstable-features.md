# Remarks on perma unstable features

## `rustc_private`

### Overview

The `rustc_private` feature allows external crates to use compiler internals.

### Using `rustc-private` with Official Toolchains

When using the `rustc_private` feature with official Rust toolchains distributed via rustup, you need to install two additional components:

1. **`rustc-dev`**: Provides compiler libraries
2. **`llvm-tools`**: Provides LLVM libraries required for linking

#### Installation Steps

Install both components using rustup:

```text
rustup component add rustc-dev llvm-tools
```

#### Common Error

Without the `llvm-tools` component, you'll encounter linking errors like:

```text
error: linking with `cc` failed: exit status: 1
  |
  = note: rust-lld: error: unable to find library -lLLVM-{version}
```

### Using `rustc-private` with Custom Toolchains

For custom-built toolchains or environments not using rustup, additional configuration is typically required:

#### Requirements

- LLVM libraries must be available in your system's library search paths
- The LLVM version must match the one used to build your Rust toolchain

#### Troubleshooting Steps

1. **Check LLVM installation**: Verify LLVM is installed and accessible
2. **Configure library paths**: You may need to set environment variables:
   ```text
   export LD_LIBRARY_PATH=/path/to/llvm/lib:$LD_LIBRARY_PATH
   ```
3. **Check version compatibility**: Ensure your LLVM version is compatible with your Rust toolchain

### Additional Resources

- [GitHub Issue #137421](https://github.com/rust-lang/rust/issues/137421): Explains that `rustc_private` linker failures often occur because `llvm-tools` is not installed
