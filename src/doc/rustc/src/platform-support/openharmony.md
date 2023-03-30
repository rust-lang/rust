# `*-unknown-linux-ohos`

**Tier: 3**

Targets for the [OpenHarmony](https://gitee.com/openharmony/docs/) operating
system.

## Target maintainers

- Amanieu d'Antras ([@Amanieu](https://github.com/Amanieu))

## Setup

The OpenHarmony SDK doesn't currently support Rust compilation directly, so
some setup is required.

First, you must obtain the OpenHarmony SDK from [this page](https://gitee.com/openharmony/docs/tree/master/en/release-notes).
Select the version of OpenHarmony you are developing for and download the "Public SDK package for the standard system".

Create the following shell scripts that wrap Clang from the OpenHarmony SDK:

`aarch64-unknown-linux-ohos-clang.sh`

```sh
#!/bin/sh
exec /path/to/ohos-sdk/linux/native/llvm/bin/clang \
  -target aarch64-linux-ohos \
  --sysroot=/path/to/ohos-sdk/linux/native/sysroot \
  -D__MUSL__ \
  "$@"
```

`aarch64-unknown-linux-ohos-clang++.sh`

```sh
#!/bin/sh
exec /path/to/ohos-sdk/linux/native/llvm/bin/clang++ \
  -target aarch64-linux-ohos \
  --sysroot=/path/to/ohos-sdk/linux/native/sysroot \
  -D__MUSL__ \
  "$@"
```

`armv7-unknown-linux-ohos-clang.sh`

```sh
#!/bin/sh
exec /path/to/ohos-sdk/linux/native/llvm/bin/clang \
  -target arm-linux-ohos \
  --sysroot=/path/to/ohos-sdk/linux/native/sysroot \
  -D__MUSL__ \
  -march=armv7-a \
  -mfloat-abi=softfp \
  -mtune=generic-armv7-a \
  -mthumb \
  "$@"
```

`armv7-unknown-linux-ohos-clang++.sh`

```sh
#!/bin/sh
exec /path/to/ohos-sdk/linux/native/llvm/bin/clang++ \
  -target arm-linux-ohos \
  --sysroot=/path/to/ohos-sdk/linux/native/sysroot \
  -D__MUSL__ \
  -march=armv7-a \
  -mfloat-abi=softfp \
  -mtune=generic-armv7-a \
  -mthumb \
  "$@"
```

Future versions of the OpenHarmony SDK will avoid the need for this process.

## Building the target

To build a rust toolchain, create a `config.toml` with the following contents:

```toml
profile = "compiler"
changelog-seen = 2

[build]
sanitizers = true
profiler = true

[target.aarch64-unknown-linux-ohos]
cc = "/path/to/aarch64-unknown-linux-ohos-clang.sh"
cxx = "/path/to/aarch64-unknown-linux-ohos-clang++.sh"
ar = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ar"
ranlib = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ranlib"
linker  = "/path/to/aarch64-unknown-linux-ohos-clang.sh"

[target.armv7-unknown-linux-ohos]
cc = "/path/to/armv7-unknown-linux-ohos-clang.sh"
cxx = "/path/to/armv7-unknown-linux-ohos-clang++.sh"
ar = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ar"
ranlib = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ranlib"
linker  = "/path/to/armv7-unknown-linux-ohos-clang.sh"
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

You will need to configure the linker to use in `~/.cargo/config`:
```toml
[target.aarch64-unknown-linux-ohos]
ar = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ar"
linker = "/path/to/aarch64-unknown-linux-ohos-clang.sh"

[target.armv7-unknown-linux-ohos]
ar = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ar"
linker = "/path/to/armv7-unknown-linux-ohos-clang.sh"
```

## Testing

Running the Rust testsuite is possible, but currently difficult due to the way
the OpenHarmony emulator is set up (no networking).

## Cross-compilation toolchains and C code

You can use the shell scripts above to compile C code for the target.
