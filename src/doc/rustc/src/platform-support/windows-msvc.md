# `*-pc-windows-msvc`

Windows MSVC targets.

**Tier 1 with host tools:**

- `aarch64-pc-windows-msvc`: Windows on ARM64.
- `i686-pc-windows-msvc`: Windows on 32-bit x86.
- `x86_64-pc-windows-msvc`: Windows on 64-bit x86.

## Target maintainers

[@ChrisDenton](https://github.com/ChrisDenton)
[@dpaoliello](https://github.com/dpaoliello)
[@lambdageek](https://github.com/lambdageek)
[@sivadeilra](https://github.com/sivadeilra)
[@wesleywiser](https://github.com/wesleywiser)

## Requirements

### OS version

Windows 10 or higher is required for client installs, Windows Server 2016 or higher is required for server installs.

### Host tooling

The minimum supported Visual Studio version is 2017 but this support is not actively tested in CI.
It is **highly** recommended to use the latest version of VS (currently VS 2022).

### Platform details

These targets fully implement the Rust standard library.

The `extern "C"` calling convention conforms to Microsoft's default calling convention for the given architecture: [`__cdecl`] on `i686`, [`x64`] on `x86_64` and [`ARM64`] on `aarch64`.

The `*-windows-msvc` targets produce PE/COFF binaries with CodeView debuginfo, the native formats used on Windows.

[`__cdecl`]: https://learn.microsoft.com/en-us/cpp/cpp/cdecl?view=msvc-170
[`x64`]: https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention?view=msvc-170
[`ARM64`]: https://learn.microsoft.com/en-us/cpp/build/arm64-windows-abi-conventions?view=msvc-170

## Building Rust programs

These targets are distributed via `rustup` and can be installed via `rustup component add [--toolchain {name}] {target}`.

For example, adding the 32-bit x86 target to the `nightly` toolchain:

```text
rustup component add --toolchain nightly i686-pc-windows-msvc
```

or adding the ARM64 target to the active toolchain:

```text
rustup component add aarch64-pc-windows-msvc
```

## Testing

There are no special requirements for testing and running this target.

## Cross-compilation toolchains and C code

Architectural cross-compilation from one Windows host to a different Windows platform is natively supported by the MSVC toolchain provided the appropriate components are selected when using the VS Installer.

Cross-compilation from a non-Windows host to a `*-windows-msvc` target _may_ be possible but is not supported.
