# External `rustc_driver`s

## `rustc_private`

### Overview

The `rustc_private` feature allows external crates to use compiler internals.

### Using `rustc_private` with official toolchains

When using the `rustc_private` feature with official Rust toolchains distributed via rustup, you need to install two additional components:

1. **`rustc-dev`**: Provides compiler libraries
2. **`llvm-tools`**: Provides LLVM libraries required for linking

#### Installation steps

Install both components using rustup:

```text
rustup component add rustc-dev llvm-tools
```

#### Common error

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

#### Troubleshooting steps

1. Verify LLVM is installed and accessible
2. Ensure that library paths are set:
   ```sh
   export LD_LIBRARY_PATH=/path/to/llvm/lib:$LD_LIBRARY_PATH
   ```
3. Ensure your LLVM version is compatible with your Rust toolchain

### Configuring `rust-analyzer` for out-of-tree projects

When developing out-of-tree projects that use `rustc_private` crates, you can configure `rust-analyzer` to recognize these crates.

#### Configuration steps

1. Configure `rust-analyzer.rustc.source` to `"discover"` in your editor settings.

   For VS Code, add to `rust_analyzer_settings.json`:
   ```json
   {
       "rust-analyzer.rustc.source": "discover"
   }
   ```

2. Add the following to the `Cargo.toml` of every crate that uses `rustc_private`:
   ```toml
   [package.metadata.rust-analyzer]
   rustc_private = true
   ```

This configuration allows `rust-analyzer` to properly recognize and provide IDE support for `rustc_private` crates in out-of-tree projects.

### Getting nightly documentation for `rustc_private`

#### Latest nightly

For the latest nightly, you can install the `rustc-docs` component and open it directly in your browser:

```sh
rustup component add rustc-docs
rustup doc --rustc-docs
```

> Note: The `rustc-docs` component is only available for recent nightly toolchains and may not be present for every nightly date. It was first introduced in [PR #75560](https://github.com/rust-lang/rust/pull/75560) (August 2020).

#### Older nightlies

If you depend on compiler internals from an older nightly, you may want to refer to the internal documentation from that particular nightly.
The only way to do this is to generate the documentation locally.
For example, to get documentation for `nightly-2025-11-08`:

Get the Git commit hash for that nightly:

```sh
rustup toolchain install nightly-2025-11-08
rustc +nightly-2025-11-08 --version --verbose
```

The output will include a `commit-hash` line identifying the exact source revision.
Check out `rust-lang/rust` at that commit, then follow the steps in [compiler documentation](../building/compiler-documenting.md).


### Additional resources

- [GitHub Issue #137421] explains that `rustc_private` linker failures often occur because `llvm-tools` is not installed

[GitHub Issue #137421]: https://github.com/rust-lang/rust/issues/137421
