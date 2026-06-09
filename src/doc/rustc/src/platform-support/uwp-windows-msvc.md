# `x86_64-uwp-windows-msvc`, `i686-uwp-windows-msvc`, `thumbv7a-uwp-windows-msvc` and `aarch64-uwp-windows-msvc`

**Tier: 3**

Windows targets for Universal Windows Platform (UWP) applications, using MSVC toolchain.

## Target maintainers

[@bdbai](https://github.com/bdbai)

## Requirements

These targets are cross-compiled with std support. The host requirement and
binary format are the same as the corresponding non-UWP targets (i.e.
`x86_64-pc-windows-msvc`, `i686-pc-windows-msvc`, `thumbv7a-pc-windows-msvc`
and `aarch64-pc-windows-msvc`).

## Building the targets

The targets can be built by enabling them for a `rustc` build, for example:

```toml
[build]
build-stage = 1
target = ["x86_64-uwp-windows-msvc", "aarch64-uwp-windows-msvc"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for these targets. To compile for
these targets, you will either need to build Rust with the targets enabled (see
"Building the targets" above), or build your own copy of `std` by using
`build-std` or similar.

Example of building a Rust project for x64 UWP using `build-std`:

```pwsh
cargo build -Z build-std=std,panic_abort --target x86_64-uwp-windows-msvc
```

## Testing

Currently there is no support to run the rustc test suite for this target.

## Cross-compilation toolchains and C code

In general, the toolchain target should match the corresponding non-UWP
targets. Beware that not all Win32 APIs behave the same way in UWP, and some
are restricted in [AppContainer](https://learn.microsoft.com/en-us/windows/win32/secauthz/appcontainer-for-legacy-applications-)
or even not available at all. If the C code being compiled happens to use any
of restricted or unavailable APIs, consider using allowed alternatives or
disable certain feature sets to avoid using them.
