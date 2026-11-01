# `*-apple-watchos`

Apple watchOS targets.

**Tier: 2 (without Host Tools)**

- `aarch64-apple-watchos`: Apple WatchOS on ARM64.
- `aarch64-apple-watchos-sim`: Apple WatchOS Simulator on ARM64.

**Tier: 3**

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

The tier 2 targets are distributed through `rustup`, and can be installed using one of:
```console
$ rustup target add aarch64-apple-watchos
$ rustup target add aarch64-apple-watchos-sim
```

See [the instructions for iOS](./apple-ios.md#building-the-target) for how to build the tier 3 targets.

## Building Rust programs

See [the instructions for iOS](./apple-ios.md#building-rust-programs).

## Testing

See [the instructions for iOS](./apple-ios.md#testing).
