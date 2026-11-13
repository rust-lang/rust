# \*-kmc-solid_\*

**Tier: 3**

[SOLID] embedded development platform by Kyoto Microcomputer Co., Ltd.

[SOLID]: https://www.kmckk.co.jp/eng/SOLID/

The target names follow this format: `$ARCH-kmc-solid_$KERNEL-$ABI`, where `$ARCH` specifies the target processor architecture, `$KERNEL` the base kernel, and `$ABI` the target ABI (optional). The following targets are currently defined:

|          Target name           | `target_arch` | `target_vendor` | `target_os`  |
|--------------------------------|---------------|-----------------|--------------|
| `aarch64-kmc-solid_asp3`       | `aarch64`     | `kmc`           | `solid_asp3` |
| `armv7a-kmc-solid_asp3-eabi`   | `arm`         | `kmc`           | `solid_asp3` |
| `armv7a-kmc-solid_asp3-eabihf` | `arm`         | `kmc`           | `solid_asp3` |

## Target Maintainers

[@kawadakk](https://github.com/kawadakk)

## Requirements

This target is cross-compiled.
A platform-provided C compiler toolchain is required, though it can be substituted by [GNU Arm Embedded Toolchain](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm) for the purpose of building Rust and functional binaries.

## Building

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["aarch64-kmc-solid_asp3"]
```

Make sure `aarch64-kmc-elf-gcc` is included in `$PATH`. Alternatively, you can use GNU Arm Embedded Toolchain by adding the following to `bootstrap.toml`:

```toml
[target.aarch64-kmc-solid_asp3]
cc = "arm-none-eabi-gcc"
```

## Cross-compilation

This target can be cross-compiled from any hosts.

## Testing

Currently there is no support to run the rustc test suite for this target.

## Building Rust programs

Building executables is not supported yet.

If `rustc` has support for that target and the library artifacts are available, then Rust static libraries can be built for that target:

```shell
$ rustc --target aarch64-kmc-solid_asp3 your-code.rs --crate-type staticlib
$ ls libyour_code.a
```

On Rust Nightly it's possible to build without the target artifacts available:

```text
cargo build -Z build-std --target aarch64-kmc-solid_asp3
```
