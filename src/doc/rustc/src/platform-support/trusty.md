# `aarch64-unknown-trusty` and `armv7-unknown-trusty`

**Tier: 3**

[Trusty] is a secure Operating System that provides a Trusted Execution
Environment (TEE) for Android.

## Target maintainers

- Nicole LeGare (@randomPoison)
- Andrei Homescu (@ahomescu)
- Chris Wailes (chriswailes@google.com)
- As a fallback trusty-dev-team@google.com can be contacted

## Requirements

These targets are cross-compiled. They have no special requirements for the host.

Support for the standard library is work-in-progress. It is expected that
they will support alloc with the default allocator, and partially support std.

Trusty uses the ELF file format.

## Building the target

The targets can be built by enabling them for a `rustc` build, for example:

```toml
[build]
build-stage = 1
target = ["aarch64-unknown-trusty", "armv7-unknown-trusty"]
```

## Building Rust programs

There is currently no supported way to build a Trusty app with Cargo. You can
follow the [Trusty build instructions] to build the Trusty kernel along with any
Rust apps that are setup in the project.

## Testing

See the [Trusty build instructions] for information on how to build Rust code
within the main Trusty project. The main project also includes infrastructure
for testing Rust applications within a QEMU emulator.

## Cross-compilation toolchains and C code

See the [Trusty build instructions] for information on how C code is built
within Trusty.

[Trusty]: https://source.android.com/docs/security/features/trusty
[Trusty build instructions]: https://source.android.com/docs/security/features/trusty/download-and-build
