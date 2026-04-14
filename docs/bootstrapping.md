# ThingOS Rust Bootstrap Notes

This document describes the **current** `cargo xtask rustc-thingos` flow.

## What It Builds Today

Today the bootstrap produces a **Linux-hosted stage-1 cross-compiler** that can
target `x86_64-unknown-thingos`.

It does **not** yet produce a ThingOS-hosted `/bin/rustc`.

That distinction matters:

- the compiler binary currently runs on `x86_64-unknown-linux-gnu`
- the compiler can emit code for `x86_64-unknown-thingos`
- the current output should be used on the developer machine, not staged into
  the ThingOS ISO

## Build Flow

```bash
just fetch-rust
just rust-apply-patches
cargo xtask rustc-thingos
```

To opt out of this build step in composite flows (`xtask iso/run/run-hdd`), set
`SKIP_RUSTC_THINGOS=1`.

Inside `cargo xtask rustc-thingos`:

1. Writes `vendor/rust/config.toml`
2. Exports `RUST_TARGET_PATH=targets/`
3. Runs:

   ```bash
   python3 vendor/rust/x.py build --stage 1 library compiler/rustc
   ```

4. Locates the produced stage1 compiler in the current bootstrap layout
5. Caches the compiler under `target/rustc-thingos/rustc`
6. Caches a reconstructed rustlib tree under `target/rustc-thingos/rustlib`

## Bootstrap Configuration

The current `xtask` generates:

```toml
[build]
build = "x86_64-unknown-linux-gnu"
host  = ["x86_64-unknown-linux-gnu"]
target = ["x86_64-unknown-linux-gnu", "x86_64-unknown-thingos"]
local-rebuild = true

[rust]
rpath = false

[target.x86_64-unknown-thingos]
sanitizers = false
profiler = false
```

That host/target split is why the resulting compiler is Linux-hosted.

## Current Artifact Layout

The current fork/bootstrap writes outputs under the repository-root `build/`
directory rather than `vendor/rust/build/`.

Important paths:

```text
build/x86_64-unknown-linux-gnu/stage1-rustc/x86_64-unknown-linux-gnu/release/rustc-main
build/x86_64-unknown-linux-gnu/stage1-rustc/x86_64-unknown-linux-gnu/release/deps/librustc_driver-*.rlib
build/x86_64-unknown-linux-gnu/stage1/lib/rustlib/x86_64-unknown-linux-gnu/...
build/x86_64-unknown-linux-gnu/stage1-std/x86_64-unknown-thingos/release/deps/*.rlib
```

The wrapper now handles that layout explicitly.

## Cached Outputs

After a successful run:

```text
target/rustc-thingos/rustc
target/rustc-thingos/rustc-wrapper
target/rustc-thingos/rustlib/
```

The cached `rustlib/` tree includes:

- host libraries copied from `stage1/lib/rustlib`
- ThingOS target `.rlib`/`.rmeta` files copied from
  `stage1-std/x86_64-unknown-thingos/release/deps`

`rustc-wrapper` is the entry point intended for Cargo/xtask use. It injects:

- `--sysroot target/rustc-thingos`
- `LD_LIBRARY_PATH=target/rustc-thingos/lib`

This is enough for the cached compiler to be reused on the developer machine.

## Important Build Invocation Detail

Raw commands like:

```bash
cargo -Z build-std=core,alloc,std,panic_abort ...
```

still use the active rustup toolchain unless the caller also points Cargo at
the cached bootstrap compiler and sysroot.

In practice:

- `cargo xtask ...` uses the cached compiler path by default (unless `SKIP_RUSTC_THINGOS=1`)
- plain `cargo ...` does not

If plain Cargo is used directly, it may try to compile the patched ThingOS
`std` against rustup's source/sysroot layout and fail with errors such as
missing `abi` or mismatched `std/sys/net` module wiring.

## ISO Staging

ISO staging is currently disabled.

Reason: the cached compiler is a Linux ELF, not a ThingOS executable. Staging
it into the image would place an unusable binary in the guest.

Re-enable staging only after the bootstrap produces a ThingOS-hosted compiler.

## Current Build Status

As revalidated on April 12, 2026:

- `x.py` completes successfully
- ThingOS target std builds successfully
- `rustc_driver` is produced as static `.rlib` artifacts in the current layout
- the remaining work is about packaging/layout and the eventual switch to a
  ThingOS-hosted compiler

See [docs/status/rustc_build.md](./status/rustc_build.md) for the current
status summary.

## Checkout Mode

`vendor/rust` is required to be a git submodule tracked by this repository.
The submodule is pinned to the `thingos-patched` branch.

If your local checkout has a legacy plain directory instead of a gitlink,
convert it before bootstrapping:

1. `rm -rf vendor/rust`
2. `just fetch-rust`

## Next Step for a True Hosted Compiler

If the goal is a compiler that runs inside ThingOS, the next major milestone is
to change bootstrap `host` from `x86_64-unknown-linux-gnu` to
`x86_64-unknown-thingos` and revalidate the compiler build in that mode.
