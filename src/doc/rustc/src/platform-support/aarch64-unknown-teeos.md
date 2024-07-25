# `aarch64-unknown-teeos`

**Tier: 3**

Target for the TEEOS operating system.

TEEOS is a mini os run in TrustZone, for trusted/security apps. The kernel of TEEOS is HongMeng/ChCore micro kernel. The libc for TEEOS is a part of musl.
It's very small that there is no RwLock, no network, no stdin, and no file system for apps in TEEOS.

Some abbreviation:

| Abbreviation | The full text | Description |
|  ----  | ----  | ---- |
| TEE | Trusted Execution Environment | ARM TrustZone divides the system into two worlds/modes -- the secure world/mode and the normal world/mode. TEE is in the secure world/mode. |
| REE | Rich Execution Environment | The normal world. for example, Linux for Android phone is in REE side. |
| TA | Trusted Application | The app run in TEE side system. |
| CA | Client Application | The progress run in REE side system. |

TEEOS is open source in progress. [MORE about](https://gitee.com/opentrustee-group)

## Target maintainers

- Petrochenkov Vadim
- Sword-Destiny

## Setup
We use OpenHarmony SDK for TEEOS.

The OpenHarmony SDK doesn't currently support Rust compilation directly, so
some setup is required.

First, you must obtain the OpenHarmony SDK from [this page](https://gitee.com/openharmony/docs/tree/master/en/release-notes).
Select the version of OpenHarmony you are developing for and download the "Public SDK package for the standard system".

Create the following shell scripts that wrap Clang from the OpenHarmony SDK:

`aarch64-unknown-teeos-clang.sh`

```sh
#!/bin/sh
exec /path/to/ohos-sdk/linux/native/llvm/bin/clang \
  -target aarch64-linux-gnu \
  "$@"
```

`aarch64-unknown-teeos-clang++.sh`

```sh
#!/bin/sh
exec /path/to/ohos-sdk/linux/native/llvm/bin/clang++ \
  -target aarch64-linux-gnu \
  "$@"
```

## Building the target

To build a rust toolchain, create a `bootstrap.toml` with the following contents:

```toml
profile = "compiler"
change-id = 115898

[build]
sanitizers = true
profiler = true
target = ["x86_64-unknown-linux-gnu", "aarch64-unknown-teeos"]
submodules = false
compiler-docs = false
extended = true

[install]
bindir = "bin"
libdir = "lib"

[target.aarch64-unknown-teeos]
cc = "/path/to/scripts/aarch64-unknown-teeos-clang.sh"
cxx = "/path/to/scripts/aarch64-unknown-teeos-clang.sh"
linker = "/path/to/scripts/aarch64-unknown-teeos-clang.sh"
ar = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ar"
ranlib = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-ranlib"
llvm-config = "/path/to/ohos-sdk/linux/native/llvm/bin/llvm-config"
```

```text
note: You need to insert "/usr/include/x86_64-linux-gnu/" into environment variable: $C_INCLUDE_PATH
 if some header files like bits/xxx.h not found.
note: You can install gcc-aarch64-linux-gnu,g++-aarch64-linux-gnu if some files like crti.o not found.
note: You may need to install libc6-dev-i386 libc6-dev if "gnu/stubs-32.h" not found.
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

You will need to configure the linker to use in `~/.cargo/config`:
```toml
[target.aarch64-unknown-teeos]
linker = "/path/to/aarch64-unknown-teeos-clang.sh" # or aarch64-linux-gnu-ld
```

## Testing

Running the Rust testsuite is not possible now.

More information about how to test CA/TA. [See here](https://gitee.com/openharmony-sig/tee_tee_dev_kit/tree/master/docs)
