# `arm64e-apple-visionos`

**Tier: 3**

Apple arm64e visionOS.

## Target maintainers

[@cypherair](https://github.com/cypherair)

## Requirements

This target is cross-compiled and supports `std`.

This target requires the visionOS SDK provided by Xcode.

The deployment target can be configured with `XROS_DEPLOYMENT_TARGET`.

## Building the target

You can build Rust with support for the target by adding it to the `target`
list in `bootstrap.toml`:
```toml
[build]
target = ["arm64e-apple-visionos"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target.
To compile for this target, you will need to build Rust with the target enabled.

## Testing

This target is not tested by Rust CI.

Binaries are intended to run on visionOS devices with `arm64e` architecture.

The visionOS simulator targets use `arm64`, not `arm64e`.

## Cross-compilation toolchains and C code

C code should be built with Apple's Clang from Xcode, using the visionOS SDK and
matching deployment target.

The Clang target is suffixed with `-xros`.