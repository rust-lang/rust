# \*-win7-windows-gnu

**Tier: 3**

Windows targets continuing support of Windows 7.

Target triples:
- `i686-win7-windows-gnu`
- `x86_64-win7-windows-gnu`

## Target maintainers

- @tbu-

## Requirements

This target supports all of core, alloc, std and test. Host
tools may also work, though those are not currently tested.

Those targets follow Windows calling convention for extern "C".

Like any other Windows target, the created binaries are in PE format.

## Building the target

You can build Rust with support for the targets by adding it to the target list in bootstrap.toml:

```toml
[build]
build-stage = 1
target = ["x86_64-win7-windows-gnu"]
```

## Building Rust programs

Rust does not ship pre-compiled artifacts for this target. To compile for this
target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy by using `build-std` or
similar.

## Testing

Created binaries work fine on Windows or Wine using native hardware. Remote
testing is possible using the `remote-test-server` described [here](https://rustc-dev-guide.rust-lang.org/tests/running.html#running-tests-on-a-remote-machine).

## Cross-compilation toolchains and C code

Compatible C code can be built with gcc's `{i686,x86_64}-w64-mingw32-gcc`.
