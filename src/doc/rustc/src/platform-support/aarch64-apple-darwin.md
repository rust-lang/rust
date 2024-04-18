# `aarch64-apple-darwin`

**Tier: 2 with host tools**

64-bit ARM-based Apple devices running macOS, Macs with Apple Silicon M1 or M2 chips. M3 is [`arm64e-apple-darwin`](platform-support/arm64e-apple-darwin.md).

## Target maintainers

- ???

## Requirements

This target supports host tools and cross-compilation. It provides the full standard library including std and alloc.

Binaries built for this target expect a Apple Silicon Mac.

### Format

The default binary format is Mach-O, the executable format used on Apple's platforms.

## Building the target

Just add it to the `target` with: 
```
rustup target add aarch64-apple-darwin
```

## Building Rust programs

Rust ships pre-compiled artifacts for this target or build your own copy of `core` by using
`build-std`.

## Testing

Binaries produced for this target can be run directly on Apple Silicon Macs natively. 
The Rust test suite can be run for this target on such hardware.

## Cross-compilation toolchains and C code

Does the target support C code? If so, what toolchain target should users use
to build compatible C code? (This may match the target triple, or it may be a
toolchain for a different target triple, potentially with specific options or
caveats.)
