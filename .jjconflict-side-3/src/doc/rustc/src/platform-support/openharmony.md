# `*-unknown-linux-ohos`

**Tier: 2 (with Host Tools)**

* aarch64-unknown-linux-ohos
* armv7-unknown-linux-ohos
* x86_64-unknown-linux-ohos

**Tier: 3**

* loongarch64-unknown-linux-ohos

Targets for the [OpenHarmony](https://gitee.com/openharmony/docs/) operating
system.

## Target maintainers

[@Amanieu](https://github.com/Amanieu)
[@cceerczw](https://github.com/cceerczw)

## Requirements

All the ohos targets of Tier 2 with host tools support all extended rust tools.
(exclude `miri`, the support of `miri` will be added soon)

### Host toolchain

The targets require a reasonably up-to-date OpenHarmony SDK on the host.

The targets support `cargo`, which require [ohos-openssl](https://github.com/ohos-rs/ohos-openssl).

`miri` isn't supported yet, since its dependencies (`libffi` and `tikv-jemalloc-sys`) don't support
compiling for the OHOS targets.

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

`x86_64-unknown-linux-ohos-clang.sh`

```sh
#!/bin/sh
exec /path/to/ohos-sdk/linux/native/llvm/bin/clang \
  -target x86_64-linux-ohos \
  --sysroot=/path/to/ohos-sdk/linux/native/sysroot \
  -D__MUSL__ \
  "$@"
```

`x86_64-unknown-linux-ohos-clang++.sh`

```sh
#!/bin/sh
exec /path/to/ohos-sdk/linux/native/llvm/bin/clang++ \
  -target x86_64-linux-ohos \
  --sysroot=/path/to/ohos-sdk/linux/native/sysroot \
  -D__MUSL__ \
  "$@"
```

Future versions of the OpenHarmony SDK will avoid the need for this process.

## Building Rust programs

Rustup ships pre-compiled artifacts for this target, which you can install with:
```sh
rustup target add aarch64-unknown-linux-ohos
rustup target add armv7-unknown-linux-ohos
rustup target add x86_64-unknown-linux-ohos
```

You will need to configure the linker to use in `~/.cargo/config.toml`:
```toml
[target.aarch64-unknown-linux-ohos]
ar = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ar"
linker = "/path/to/aarch64-unknown-linux-ohos-clang.sh"

[target.armv7-unknown-linux-ohos]
ar = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ar"
linker = "/path/to/armv7-unknown-linux-ohos-clang.sh"

[target.x86_64-unknown-linux-ohos]
ar = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ar"
linker = "/path/to/x86_64-unknown-linux-ohos-clang.sh"
```

## Building the target from source

Instead of using `rustup`, you can instead build a rust toolchain from source.
Create a `bootstrap.toml` with the following contents:

```toml
profile = "compiler"
change-id = 115898

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

[target.x86_64-unknown-linux-ohos]
cc = "/path/to/x86_64-unknown-linux-ohos-clang.sh"
cxx = "/path/to/x86_64-unknown-linux-ohos-clang++.sh"
ar = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ar"
ranlib = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ranlib"
linker  = "/path/to/x86_64-unknown-linux-ohos-clang.sh"
```

## Testing

Running the Rust testsuite is possible, but currently difficult due to the way
the OpenHarmony emulator is set up (no networking).

## Cross-compilation toolchains and C code

You can use the shell scripts above to compile C code for the target.
