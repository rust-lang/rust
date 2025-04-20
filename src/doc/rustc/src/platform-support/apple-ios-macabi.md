# `*-apple-ios-macabi`

Apple Mac Catalyst targets.

**Tier: 2 (without Host Tools)**

- `aarch64-apple-ios-macabi`: Mac Catalyst on ARM64.
- `x86_64-apple-ios-macabi`: Mac Catalyst on 64-bit x86.

## Target maintainers

[@badboy](https://github.com/badboy)
[@BlackHoleFox](https://github.com/BlackHoleFox)
[@madsmtm](https://github.com/madsmtm)

## Requirements

These targets are cross-compiled, and require the corresponding macOS SDK
(`MacOSX.sdk`) which contain `./System/iOSSupport` headers to allow linking to
iOS-specific headers, as provided by Xcode 11 or higher.

The path to the SDK can be passed to `rustc` using the common `SDKROOT`
environment variable, or will be inferred when compiling on host macOS using
roughly the same logic as `xcrun --sdk macosx --show-sdk-path`.

### OS version

The minimum supported version is iOS 13.1 on x86 and 14.0 on Aarch64.

This can be raised per-binary by changing the deployment target. `rustc`
respects the common environment variables used by Xcode to do so, in this
case `IPHONEOS_DEPLOYMENT_TARGET`.

## Building the target

The targets are distributed through `rustup`, and can be installed using one
of:
```console
$ rustup target add aarch64-apple-ios-macabi
$ rustup target add x86_64-apple-ios-macabi
```

### Sanitizers

Due to CMake having poor support for Mac Catalyst, sanitizer runtimes are not
currently available, see [#129069].

[#129069]: https://github.com/rust-lang/rust/issues/129069

## Building Rust programs

Rust programs can be built for these targets by specifying `--target`, if
`rustc` has been built with support for them. For example:

```console
$ rustc --target aarch64-apple-ios-macabi your-code.rs
```

## Testing

Mac Catalyst binaries can be run directly on macOS 10.15 Catalina or newer.

x86 binaries can be run on Apple Silicon by using Rosetta.

Note that using certain UIKit functionality requires the binary to be bundled.
