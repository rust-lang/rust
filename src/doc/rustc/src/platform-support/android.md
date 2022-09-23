# *-linux-android and *-linux-androideabi

**Tier: 2**

[Android] is a mobile operating system built on top of the Linux kernel.

[Android]: https://source.android.com/

## Target maintainers

- Chris Wailes ([@chriswailes](https://github.com/chriswailes))
- Matthew Maurer ([@maurer](https://github.com/maurer))
- Martin Geisler ([@mgeisler](https://github.com/mgeisler))

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

[Android NDK]: https://developer.android.com/ndk/downloads

A list of all supported targets can be found
[here](https://doc.rust-lang.org/rustc/platform-support.html)
