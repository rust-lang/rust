# riscv64gc-linux-android

**Tier: 3**

Android target similar to "x86_64-linux-android" and "aarch64-linux-android".

## Target maintainers

- [@Mao Han](https://github.com/MaoHan001)

## Requirements

The requirements for android rust toolchian can be downloaded with:
repo init -u https://android.googlesource.com/platform/manifest -b rust-toolchain

## Building the target

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["riscv64gc-linux-android"]
```

Make sure your C compiler is included in `$PATH`, then add it to the `config.toml`:

```toml
[target.riscv64gc-linux-android]
cc="clang-riscv64-linux-android"
ar="llvm-ar"
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

## Testing
Check adb can list the attached device
$ adb devices
List of devices attached
emulator-5580   device

Test assembly with stage 3 compiler
$ ./x.py test src/test/assembly --stage 3 --target riscv64gc-linux-android

## Cross-compilation toolchains and C code

Compatible C code can be built with Clang's `riscv64-linux-android` targets as long as LLVM based C toolchains are used.
- [clang-riscv64](https://github.com/riscv-android-src/platform-prebuilts-clang-host-linux-x86/tree/riscv64-android-12.0.0_dev)
