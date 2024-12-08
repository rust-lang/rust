# `*-apple-ios`

Apple iOS / iPadOS targets.

**Tier: 2 (without Host Tools)**

- `aarch64-apple-ios`: Apple iOS on ARM64.
- `aarch64-apple-ios-sim`: Apple iOS Simulator on ARM64.
- `x86_64-apple-ios`: Apple iOS Simulator on 64-bit x86.

**Tier: 3**

- `armv7s-apple-ios`: Apple iOS on Armv7-A.
- `i386-apple-ios`: Apple iOS Simulator on 32-bit x86.

## Target maintainers

- [@badboy](https://github.com/badboy)
- [@deg4uss3r](https://github.com/deg4uss3r)
- [@madsmtm](https://github.com/madsmtm)

## Requirements

These targets are cross-compiled, and require the corresponding iOS SDK
(`iPhoneOS.sdk` or `iPhoneSimulator.sdk`), as provided by Xcode. To build the
ARM64 targets, Xcode 12 or higher is required.

The path to the SDK can be passed to `rustc` using the common `SDKROOT`
environment variable.

### OS version

The minimum supported version is iOS 10.0.

This can be raised per-binary by changing the deployment target. `rustc`
respects the common environment variables used by Xcode to do so, in this
case `IPHONEOS_DEPLOYMENT_TARGET`.

## Building the target

The tier 2 targets are distributed through `rustup`, and can be installed
using one of:
```console
$ rustup target add aarch64-apple-ios
$ rustup target add aarch64-apple-ios-sim
$ rustup target add x86_64-apple-ios
```

The tier 3 targets can be built by enabling them for a `rustc` build in
`config.toml`, by adding, for example:

```toml
[build]
target = ["armv7s-apple-ios", "i386-apple-ios"]
```

Using the unstable `-Zbuild-std` with a nightly Cargo may also work.

## Building Rust programs

Rust programs can be built for these targets by specifying `--target`, if
`rustc` has been built with support for them. For example:

```console
$ rustc --target aarch64-apple-ios your-code.rs
```

## Testing

There is no support for running the Rust or standard library testsuite at the
moment. Testing has mostly been done manually with builds of static libraries
embedded into applications called from Xcode or a simulator.

It hopefully will be possible to improve this in the future.
