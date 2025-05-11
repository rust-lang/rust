# *-linux-android and *-linux-androideabi

**Tier: 2**

[Android] is a mobile operating system built on top of the Linux kernel.

[Android]: https://source.android.com/

## Target maintainers

[@chriswailes](https://github.com/chriswailes)
[@maurer](https://github.com/maurer)
[@mgeisler](https://github.com/mgeisler)

## Requirements

This target is cross-compiled from a host environment. Development may be done
from the [source tree] or using the Android NDK.

[source tree]: https://source.android.com/docs/setup/build/downloading

Android targets support std. Generated binaries use the ELF file format.

## NDK/API Update Policy

Rust will support the most recent Long Term Support (LTS) Android Native
Development Kit (NDK).  By default Rust will support all API levels supported
by the NDK, but a higher minimum API level may be required if deemed necessary.

## Building the target

To build Rust binaries for Android you'll need a copy of the most recent LTS
edition of the [Android NDK].  Supported Android targets are:

* aarch64-linux-android
* arm-linux-androideabi
* armv7-linux-androideabi
* i686-linux-android
* thumbv7neon-linux-androideabi
* x86_64-linux-android

The riscv64-linux-android target is supported as a Tier 3 target.

[Android NDK]: https://developer.android.com/ndk/downloads

A list of all supported targets can be found
[here](../platform-support.html)

## Architecture Notes

### riscv64-linux-android

Currently the `riscv64-linux-android` target requires the following architecture features/extensions:

* `a` (atomics)
* `d` (double-precision floating-point)
* `c` (compressed instruction set)
* `f` (single-precision floating-point)
* `m` (multiplication and division)
* `v` (vector)
* `Zba` (address calculation instructions)
* `Zbb` (base instructions)
* `Zbs` (single-bit instructions)

### aarch64-linux-android on Nightly compilers

As soon as `-Zfixed-x18` compiler flag is supplied, the [`ShadowCallStack` sanitizer](https://releases.llvm.org/7.0.1/tools/clang/docs/ShadowCallStack.html)
instrumentation is also made available by supplying the second compiler flag `-Zsanitizer=shadow-call-stack`.
