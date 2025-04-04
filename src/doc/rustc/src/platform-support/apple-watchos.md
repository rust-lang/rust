# `*-apple-watchos`

Apple watchOS targets.

**Tier: 3**

- `aarch64-apple-watchos`: Apple WatchOS on ARM64.
- `aarch64-apple-watchos-sim`: Apple WatchOS Simulator on ARM64.
- `x86_64-apple-watchos-sim`: Apple WatchOS Simulator on 64-bit x86.
- `arm64_32-apple-watchos`: Apple WatchOS on Arm 64_32.
- `armv7k-apple-watchos`: Apple WatchOS on Armv7k.

## Target maintainers

[@deg4uss3r](https://github.com/deg4uss3r)
[@vladimir-ea](https://github.com/vladimir-ea)
[@leohowell](https://github.com/leohowell)
[@madsmtm](https://github.com/madsmtm)

## Requirements

These targets are cross-compiled, and require the corresponding watchOS SDK
(`WatchOS.sdk` or `WatchSimulator.sdk`), as provided by Xcode. To build the
ARM64 targets, Xcode 12 or higher is required.

The path to the SDK can be passed to `rustc` using the common `SDKROOT`
environment variable, or will be inferred when compiling on host macOS using
roughly the same logic as `xcrun --sdk watchos --show-sdk-path`.

### OS version

The minimum supported version is watchOS 5.0.

This can be raised per-binary by changing the deployment target. `rustc`
respects the common environment variables used by Xcode to do so, in this
case `WATCHOS_DEPLOYMENT_TARGET`.

## Building the target

The targets can be built by enabling them for a `rustc` build in
`bootstrap.toml`, by adding, for example:

```toml
[build]
build-stage = 1
target = ["aarch64-apple-watchos", "aarch64-apple-watchos-sim"]
```

Using the unstable `-Zbuild-std` with a nightly Cargo may also work.

## Building Rust programs

Rust programs can be built for these targets by specifying `--target`, if
`rustc` has been built with support for them. For example:

```console
$ rustc --target aarch64-apple-watchos-sim your-code.rs
```

## Testing

There is no support for running the Rust or standard library testsuite at the
moment. Testing has mostly been done manually with builds of static libraries
embedded into applications called from Xcode or a simulator.

It hopefully will be possible to improve this in the future.
