# \*-win7-windows-msvc

**Tier: 3**

Windows targets continuing support of Windows 7.

Target triples:
- `i686-win7-windows-msvc`
- `x86_64-win7-windows-msvc`

## Target maintainers

[@roblabla](https://github.com/roblabla)

## Requirements

This target supports all of core, alloc, std and test. This is automatically
tested every night on private infrastructure hosted by the maintainer. Host
tools may also work, though those are not currently tested.

Those targets follow Windows calling convention for extern "C".

Like any other Windows target, the created binaries are in PE format.

## Building the target

You can build Rust with support for the targets by adding it to the target list in bootstrap.toml:

```toml
[build]
build-stage = 1
target = ["x86_64-win7-windows-msvc"]
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

Compatible C code can be built with either MSVC's `cl.exe` or LLVM's clang-cl.

Cross-compilation is possible using clang-cl/lld-link. It also requires the
Windows SDK, which can be acquired using [`xwin`](https://github.com/Jake-Shadle/xwin).

- Install `clang-cl` and `lld-link` on your machine, and make sure they are in
  your $PATH.
- Install `xwin`: `cargo install xwin`
- Use `xwin` to install the Windows SDK: `xwin splat --output winsdk`
- Create an `xwin-lld-link` script with the following content:

  ```bash
  #!/usr/bin/env bash
  set -e
  XWIN=/path/to/winsdk
  lld-link "$@" /libpath:$XWIN/crt/lib/x86_64 /libpath:$XWIN/sdk/lib/um/x86_64 /libpath:$XWIN/sdk/lib/ucrt/x86_64
  ```

- Create an `xwin-clang-cl` script with the following content:

  ```bash
  #!/usr/bin/env bash
  set -e
  XWIN=/path/to/winsdk
  clang-cl /imsvc "$XWIN/crt/include" /imsvc "$XWIN/sdk/include/ucrt" /imsvc "$XWIN/sdk/include/um" /imsvc "$XWIN/sdk/include/shared" --target="x86_64-pc-windows-msvc" "$@"
  ```

- In your bootstrap.toml, add the following lines:

  ```toml
  [target.x86_64-win7-windows-msvc]
  linker = "path/to/xwin-lld-link"
  cc = "path/to/xwin-clang-cl"
  ```

You should now be able to cross-compile the Rust std, and any rust program.
