# aarch64-apple-visionos\*

-   aarch64-apple-visionos
-   aarch64-apple-visionos-sim

**Tier: 3**

Apple visionOS targets:

-   Apple visionOS on arm64
-   Apple visionOS Simulator on arm64

## Target maintainers

-   [@agg23](https://github.com/agg23)
-   [@madsmtm](https://github.com/madsmtm)

## Requirements

These targets are cross-compiled.
To build these targets Xcode 15 or higher on macOS is required, along with LLVM 18.

## Building the target

The targets can be built by enabling them for a `rustc` build, for example:

```toml
[build]
build-stage = 1
target = ["aarch64-apple-visionos-sim"]
```

## Building Rust programs

_Note: Building for this target requires the corresponding visionOS SDK, as provided by Xcode 15+._

Rust programs can be built for these targets, if `rustc` has been built with support for them, for example:

```text
rustc --target aarch64-apple-visionos-sim your-code.rs
```

## Testing

There is no support for running the Rust testsuite on visionOS or the simulators.

There is no easy way to run simple programs on visionOS or the visionOS simulators. Static library builds can be embedded into visionOS applications.

## Cross-compilation toolchains and C code

This target can be cross-compiled from x86_64 or aarch64 macOS hosts.

Other hosts are not supported for cross-compilation, but might work when also providing the required Xcode SDK.
