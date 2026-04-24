# `arm64e-apple-visionos`

**Tier: 3**

ARM64e visionOS (1.0+)

## Target maintainers

[@arttet](https://github.com/arttet)

## Requirements

See the docs on [`*-apple-visionos`](apple-visionos.md) for general visionOS
requirements.

## Building the target

You can build Rust with support for the target by adding it to the `target`
list in `bootstrap.toml`:

```toml
[build]
target = ["arm64e-apple-visionos"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target.
To compile for this target, you will need to build Rust with the target enabled
(see [Building the target](#building-the-target) above).

## Testing

The target does support running binaries on visionOS platforms with `arm64e`
architecture.

## Cross-compilation toolchains and C code

The target does support `C` code.
To build compatible `C` code, you have to use Xcode with the same compiler and
flags.
