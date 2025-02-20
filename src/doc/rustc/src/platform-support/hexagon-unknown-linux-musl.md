# `hexagon-unknown-linux-musl`

**Tier: 3**

Target for cross-compiling Linux user-mode applications targeting the Hexagon
DSP architecture.

| Target                   | Descriptions                              |
| ------------------------ | ----------------------------------------- |
| hexagon-unknown-linux-musl | Hexagon 32-bit Linux |

## Target maintainers

- [Brian Cain](https://github.com/androm3da), `bcain@quicinc.com`

## Requirements
The target is cross-compiled. This target supports `std`.  By default, code
generated with this target should run on Hexagon DSP hardware.

- `-Ctarget-cpu=hexagonv73` adds support for instructions defined up to Hexagon V73.

Binaries can be run using QEMU user emulation. On Debian-based systems, it should be
sufficient to install the package `qemu-user-static` to be able to run simple static
binaries:

```text
# apt install qemu-user-static
# qemu-hexagon-static ./hello
```

In order to build linux programs with Rust, you will require a linker capable
of targeting hexagon.  You can use `clang`/`lld` from the [hexagon toolchain
using exclusively public open source repos](https://github.com/quic/toolchain_for_hexagon/releases).

Also included in that toolchain is the C library that can be used when creating
dynamically linked executables.

```text
# /opt/clang+llvm-18.1.0-cross-hexagon-unknown-linux-musl/x86_64-linux-gnu/bin/qemu-hexagon -L /opt/clang+llvm-18.1.0-cross-hexagon-unknown-linux-musl/x86_64-linux-gnu/target/hexagon-unknown-linux-musl/usr/ ./hello
```

## Building the target
Because it is Tier 3, rust does not yet ship pre-compiled artifacts for this
target.

Therefore, you can build Rust with support for the target by adding it to the
target list in `bootstrap.toml`, a sample configuration is shown below.

```toml
[build]
target = ["hexagon-unknown-linux-musl"]

[target.hexagon-unknown-linux-musl]

cc = "hexagon-unknown-linux-musl-clang"
cxx = "hexagon-unknown-linux-musl-clang++"
linker = "hexagon-unknown-linux-musl-clang"
ar = "hexagon-unknown-linux-musl-ar"
ranlib = "hexagon-unknown-linux-musl-ranlib"
musl-root = "/opt/clang+llvm-18.1.0-cross-hexagon-unknown-linux-musl/x86_64-linux-gnu/target/hexagon-unknown-linux-musl/usr"
llvm-libunwind = 'in-tree'
qemu-rootfs = "/opt/clang+llvm-18.1.0-cross-hexagon-unknown-linux-musl/x86_64-linux-gnu/target/hexagon-unknown-linux-musl/usr"
```


## Testing

Currently there is no support to run the rustc test suite for this target.


## Building Rust programs

Download and install the hexagon open source toolchain from https://github.com/quic/toolchain_for_hexagon/releases

The following `.cargo/config` is needed inside any project directory to build
for the Hexagon Linux target:

```toml
[build]
target = "hexagon-unknown-linux-musl"

[target.hexagon-unknown-linux-musl]
linker = "hexagon-unknown-linux-musl-clang"
ar = "hexagon-unknown-linux-musl-ar"
runner = "qemu-hexagon -L /opt/clang+llvm-18.1.0-cross-hexagon-unknown-linux-musl/x86_64-linux-gnu/target/hexagon-unknown-linux-musl/usr"
```

Edit the "runner" in `.cargo/config` to point to the path to your toolchain's
C library.

```text
...
runner = "qemu-hexagon -L /path/to/my/inst/clang+llvm-18.1.0-cross-hexagon-unknown-linux-musl/x86_64-linux-gnu/target/hexagon-unknown-linux-musl/usr"
...
```

Build/run your rust program with `qemu-hexagon` in your `PATH`:

```text
export PATH=/path/to/my/inst/clang+llvm-18.1.0-cross-hexagon-unknown-linux-musl/x86_64-linux-gnu/bin/:$PATH
cargo run -Zbuild-std -Zbuild-std-features=llvm-libunwind
```
