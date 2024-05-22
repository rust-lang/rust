# `*-apple-ios-macabi`

Apple Mac Catalyst targets.

**Tier: 3**

- `aarch64-apple-ios-macabi`: Mac Catalyst on ARM64.
- `x86_64-apple-ios-macabi`: Mac Catalyst on 64-bit x86.

## Target maintainers

- [@madsmtm](https://github.com/madsmtm)

## Requirements

These targets are cross-compiled, and require the corresponding macOS SDK
(`MacOSX.sdk`) which contain `./System/iOSSupport` headers to allow linking to
iOS-specific headers, as provided by Xcode 11 or higher.

The path to the SDK can be passed to `rustc` using the common `SDKROOT`
environment variable.

### OS version

The minimum supported version is iOS 13.1.

This can be raised per-binary by changing the deployment target. `rustc`
respects the common environment variables used by Xcode to do so, in this
case `IPHONEOS_DEPLOYMENT_TARGET`.

## Building the target

The targets can be built by enabling them for a `rustc` build in
`config.toml`, by adding, for example:

```toml
[build]
target = ["aarch64-apple-ios-macabi", "x86_64-apple-ios-macabi"]
```

Using the unstable `-Zbuild-std` with a nightly Cargo may also work.

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
