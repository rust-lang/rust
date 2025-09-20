# `*-apple-visionos`

Apple visionOS / xrOS targets.

**Tier: 3**

- `aarch64-apple-visionos`: Apple visionOS on arm64.
- `aarch64-apple-visionos-sim`: Apple visionOS Simulator on arm64.

## Target maintainers

[@agg23](https://github.com/agg23)
[@madsmtm](https://github.com/madsmtm)

## Requirements

These targets are cross-compiled, and require the corresponding visionOS SDK
(`XROS.sdk` or `XRSimulator.sdk`), as provided by Xcode 15 or newer.

The path to the SDK can be passed to `rustc` using the common `SDKROOT`
environment variable, or will be inferred when compiling on host macOS using
roughly the same logic as `xcrun --sdk xros --show-sdk-path`.

### OS version

The minimum supported version is visionOS 1.0.

This can be raised per-binary by changing the deployment target. `rustc`
respects the common environment variables used by Xcode to do so, in this
case `XROS_DEPLOYMENT_TARGET`.

## Building the target

The targets can be built by enabling them for a `rustc` build in
`bootstrap.toml`, by adding, for example:

```toml
[build]
target = ["aarch64-apple-visionos", "aarch64-apple-visionos-sim"]
```

Using the unstable `-Zbuild-std` with a nightly Cargo may also work.

Note: Currently, a newer version of `libc` and `cc` may be required, this will
be fixed in [#124560](https://github.com/rust-lang/rust/pull/124560).

## Building Rust programs

See [the instructions for iOS](./apple-ios.md#building-rust-programs).

## Testing

See [the instructions for iOS](./apple-ios.md#testing).

## Cross-compilation toolchains and C code

The Clang target is suffixed with `-xros` for historical reasons.

LLVM 18 or newer is required to build this target.
