# `aarch64-apple-darwin`

**Tier: 2 with host tools**

64-bit ARM-based Apple devices running macOS, Macs with M1 or M2 Apple Silicon chips. M3 or higher is [`arm64e-apple-darwin`](platform-support/arm64e-apple-darwin.md).

## Target maintainers

- ???

## Requirements

This target supports host tools and cross-compilation. It provides the full standard library including std and alloc.

Binaries built for this target expect a Apple Silicon Mac on macOS 11.0 Big Sur+.

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

This target supports C code. To build compatible C code, you should use a toolchain targeting aarch64-apple-darwin, such as Xcode on a Mac or GCC.
The toolchain target triple matches this Rust target triple.
