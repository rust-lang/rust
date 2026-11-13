# `arm64e-apple-tvos`

**Tier: 3**

ARM64e tvOS (10.0+)

## Target maintainers

[@arttet](https://github.com/arttet)

## Requirements

This target is cross-compiled and supports `std`.
To build this target Xcode 12 or higher on macOS is required.

## Building the target

You can build Rust with support for the targets by adding it to the `target` list in `bootstrap.toml`:

```toml
[build]
target = ["arm64e-apple-tvos"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target.
To compile for this target, you will need to build Rust with the target enabled (see [Building the target](#building-the-target) above).

## Testing

The target does support running binaries on tvOS platforms with `arm64e` architecture.

## Cross-compilation toolchains and C code

The targets do support `C` code.
To build compatible `C` code, you have to use XCode with the same compiler and flags.
