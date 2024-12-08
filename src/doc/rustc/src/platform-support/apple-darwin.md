# `*-apple-darwin`

Apple macOS targets.

**Tier: 1**

- `x86_64-apple-darwin`: macOS on 64-bit x86.
- `aarch64-apple-darwin`: macOS on ARM64 (M1-family or later Apple Silicon CPUs).

## Target maintainers

- [@thomcc](https://github.com/thomcc)
- [@madsmtm](https://github.com/madsmtm)

## Requirements

### OS version

The minimum supported version is macOS 10.12 Sierra on x86, and macOS 11.0 Big
Sur on ARM64.

This version can be raised per-binary by changing the [deployment target],
which might yield more performance optimizations. `rustc` respects the common
environment variables used by Xcode to do so, in this case
`MACOSX_DEPLOYMENT_TARGET`.

The current default deployment target for `rustc` can be retrieved with
[`rustc --print=deployment-target`][rustc-print].

[deployment target]: https://developer.apple.com/library/archive/documentation/DeveloperTools/Conceptual/cross_development/Configuring/configuring.html
[rustc-print]: ../command-line-arguments.md#option-print

### Binary format

The default binary format is Mach-O, the executable format used on Apple's
platforms.

## Building

These targets are distributed through `rustup`, and otherwise require no
special configuration.

## Testing

There are no special requirements for testing and running this target.

x86 binaries can be run on Apple Silicon by using Rosetta.

## Cross-compilation toolchains and C code

Cross-compilation of these targets are supported using Clang, but may require
Xcode or the macOS SDK (`MacOSX.sdk`) to be available to compile C code and
to link.

The Clang target is suffixed with `-macosx`. Clang's `-darwin` target refers
to Darwin platforms in general (macOS/iOS/tvOS/watchOS/visionOS), and requires
the `-mmacosx-version-min=...`, `-miphoneos-version-min=...` or similar flags
to disambiguate.

The path to the SDK can be passed to `rustc` using the common `SDKROOT`
environment variable.
