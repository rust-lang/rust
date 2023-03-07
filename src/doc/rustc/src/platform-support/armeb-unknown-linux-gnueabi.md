# armeb-unknown-linux-gnueabi
**Tier: 3**

Target for cross-compiling Linux user-mode applications targeting the ARM BE8 architecture.

## Overview
BE8 architecture retains the same little-endian ordered code-stream used by conventional little endian ARM systems, however the data accesses are in big-endian. BE8 is used primarily in high-performance networking applications where the ability to read packets in their native "Network Byte Order" is important (many network protocols transmit data in big-endian byte order for their wire formats).

## History
BE8 architecture is the default big-endian architecture for ARM since [ARMv6](https://developer.arm.com/documentation/101754/0616/armlink-Reference/armlink-Command-line-Options/--be8?lang=en). It's predecessor, used for ARMv4 and ARMv5 devices was [BE32](https://developer.arm.com/documentation/dui0474/j/linker-command-line-options/--be32). On ARMv6 architecture, endianness can be configured via [system registers](https://developer.arm.com/documentation/ddi0290/g/unaligned-and-mixed-endian-data-access-support/mixed-endian-access-support/interaction-between-the-bus-protocol-and-the-core-endianness). However, BE32 was withdrawn for [ARMv7](https://developer.arm.com/documentation/ddi0406/cb/Appendixes/Deprecated-and-Obsolete-Features/Obsolete-features/Support-for-BE-32-endianness-model) onwards.

## Target Maintainers
* [@WorksButNotTested](https://github.com/WorksButNotTested)

## Requirements
The target is cross-compiled. This target supports `std` in the normal way (indeed only nominal changes are required from the standard ARM configuration).

## Target definition
The target definition can be seen [here](https://github.com/rust-lang/rust/tree/master/compiler/rustc_target/src/spec/armeb_unknown_linux_gnueabi.rs). In particular, it should be noted that the `features` specify that this target is built for the ARMv8 core. Though this can likely be modified as required.

## Building the target
Because it is Tier 3, rust does not yet ship pre-compiled artifacts for this target.

Therefore, you can build Rust with support for the target by adding it to the target list in config.toml, a sample configuration is shown below. It is expected that the user already have a working GNU compiler toolchain and update the paths accordingly.

```toml
[llvm]
download-ci-llvm = false
optimize = true
ninja = true
targets = "ARM;X86"
clang = false

[build]
target = ["x86_64-unknown-linux-gnu", "armeb-unknown-linux-gnueabi"]
docs = false
docs-minification = false
compiler-docs = false
[install]
prefix = "/home/user/x-tools/rust/"

[rust]
debug-logging=true
backtrace = true
incremental = true

[target.x86_64-unknown-linux-gnu]

[dist]

[target.armeb-unknown-linux-gnueabi]
cc = "/home/user/x-tools/armeb-unknown-linux-gnueabi/bin/armeb-unknown-linux-gnueabi-gcc"
cxx = "/home/user/x-tools/armeb-unknown-linux-gnueabi/bin/armeb-unknown-linux-gnueabi-g++"
ar = "/home/user/x-tools/armeb-unknown-linux-gnueabi/bin/armeb-unknown-linux-gnueabi-ar"
ranlib = "/home/user/x-tools/armeb-unknown-linux-gnueabi/bin/armeb-unknown-linux-gnueabi-ranlib"
linker = "/home/user/x-tools/armeb-unknown-linux-gnueabi/bin/armeb-unknown-linux-gnueabi-gcc"
llvm-config = "/home/user/x-tools/clang/bin/llvm-config"
llvm-filecheck = "/home/user/x-tools/clang/bin/FileCheck"
```

## Building Rust programs

The following `.cargo/config` is needed inside any project directory to build for the BE8 target:

```toml
[build]
target = "armeb-unknown-linux-gnueabi"

[target.armeb-unknown-linux-gnueabi]
linker = "armeb-unknown-linux-gnueabi-gcc"
```

Note that it is expected that the user has a suitable linker from the GNU toolchain.
