# *-apple-watchos
- arm64_32-apple-watchos
- armv7k-apple-watchos
- aarch64-apple-watchos-sim
- x86_64-apple-watchos-sim

**Tier: 3**

Apple WatchOS targets:
- Apple WatchOS on Arm 64_32
- Apple WatchOS on Arm v7k
- Apple WatchOS Simulator on arm64
- Apple WatchOS Simulator on x86_64

## Target maintainers

* [@deg4uss3r](https://github.com/deg4uss3r)
* [@vladimir-ea](https://github.com/vladimir-ea)

## Requirements

These targets are cross-compiled.
To build these targets Xcode 12 or higher on macOS is required.

## Building the target

The targets can be built by enabling them for a `rustc` build, for example:

```toml
[build]
build-stage = 1
target = ["aarch64-apple-watchos-sim"]
```

## Building Rust programs

*Note: Building for this target requires the corresponding WatchOS SDK, as provided by Xcode 12+.*

Rust programs can be built for these targets, if `rustc` has been built with support for them, for example:

```text
rustc --target aarch64-apple-watchos-sim your-code.rs
```

## Testing

There is no support for running the Rust testsuite on WatchOS or the simulators.

There is no easy way to run simple programs on WatchOS or the WatchOS simulators. Static library builds can be embedded into WatchOS applications.

## Cross-compilation toolchains and C code

This target can be cross-compiled from x86_64 or aarch64 macOS hosts.

Other hosts are not supported for cross-compilation, but might work when also providing the required Xcode SDK.
