# `arm64e-apple-darwin`

**Tier: 3 (with Host Tools)**

ARM64e macOS (11.0+, Big Sur+)

## Target maintainers

[@arttet](https://github.com/arttet)

## Requirements

Target for `macOS` on late-generation `M` series Apple chips.

See the docs on [`*-apple-darwin`](apple-darwin.md) for general macOS requirements.

## Building the target

You can build Rust with support for the targets by adding it to the `target` list in `bootstrap.toml`:

```toml
[build]
target = ["arm64e-apple-darwin"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target.
To compile for this target, you will need to build Rust with the target enabled (see [Building the target](#building-the-target) above).

## Testing

The target does support running binaries on macOS platforms with `arm64e` architecture.

## Cross-compilation toolchains and C code

The targets do support `C` code.
To build compatible `C` code, you have to use XCode with the same compiler and flags.
