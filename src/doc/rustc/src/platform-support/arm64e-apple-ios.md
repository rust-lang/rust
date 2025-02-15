# `arm64e-apple-ios`

**Tier: 3**

ARM64e iOS (14.0+)

## Target maintainers

- Artyom Tetyukhin ([@arttet](https://github.com/arttet))

## Requirements

See the docs on [`*-apple-ios`](apple-ios.md) for general iOS requirements.

## Building the target

You can build Rust with support for the targets by adding it to the `target` list in `bootstrap.toml`:

```toml
[build]
target = ["arm64e-apple-ios"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target.
To compile for this target, you will need to build Rust with the target enabled (see [Building the target](#building-the-target) above).

## Testing

The target does support running binaries on iOS platforms with `arm64e` architecture.

## Cross-compilation toolchains and C code

The targets do support `C` code.
To build compatible `C` code, you have to use XCode with the same compiler and flags.
