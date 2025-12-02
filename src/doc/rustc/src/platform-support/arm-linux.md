# Arm Linux support in Rust

The Arm Architecture has been around since the mid-1980s, going through nine
major revisions, many minor revisions, and spanning both 32-bit and 64-bit
architectures. This page covers 32-bit Arm platforms that run some form of
Linux (but not Android). Those targets are:

* `arm-unknown-linux-gnueabi`
* `arm-unknown-linux-gnueabihf`
* `arm-unknown-linux-musleabi`
* `arm-unknown-linux-musleabihf`
* [`armeb-unknown-linux-gnueabi`](armeb-unknown-linux-gnueabi.md)
* `armv4t-unknown-linux-gnueabi`
* [`armv5te-unknown-linux-gnueabi`](armv5te-unknown-linux-gnueabi.md)
* `armv5te-unknown-linux-musleabi`
* `armv5te-unknown-linux-uclibceabi`
* [`armv7-unknown-linux-gnueabi`](armv7-unknown-linux-gnueabi.md)
* [`armv7-unknown-linux-gnueabihf`](armv7-unknown-linux-gnueabi.md)
* `armv7-unknown-linux-musleabi`
* `armv7-unknown-linux-musleabihf`
* `armv7-unknown-linux-ohos`
* [`armv7-unknown-linux-uclibceabi`](armv7-unknown-linux-uclibceabi.md)
* [`armv7-unknown-linux-uclibceabihf`](armv7-unknown-linux-uclibceabihf.md)
* `thumbv7neon-unknown-linux-gnueabihf`
* `thumbv7neon-unknown-linux-musleabihf`

Some of these targets have dedicated pages and some do not. This is largely
due to historical accident, or the enthusiasm of the maintainers. This
document attempts to cover all the targets, but only in broad terms.

To make sense of this list, the architecture and ABI component of the
`<architecture>-unknown-linux-<abi>` tuple will be discussed separately.

The second part of the tuple is `unknown` because these systems don't come
from any one specific vendor (like `powerpc-ibm-aix` or
`aarch64-apple-darwin`). The third part is `linux`, because this page only
discusses Linux targets.

## Architecture Component

* `arm`
* `armeb`
* `armv4t`
* `armv5te`
* `armv7`
* `thumbv7neon`

The architecture component simply called `arm` corresponds to the Armv6
architecture - that is, version 6 of the Arm Architecture as defined in
version 6 of the Arm Architecture Reference Manual (the Arm ARM). This was the
last 'legacy' release of the Arm architecture, before they split into
Application, Real-Time, and Microcontroller profiles (leading to Armv7-A,
Armv7-R and Armv7-M). Processors that implement the Armv6 architecture include
the ARM1176JZF-S, as found in BCM2835 SoC that powers the Raspberry Pi Zero.
Arm processors are generally fairly backwards compatible, especially for
user-mode code, so code compiled for the `arm` architecture should also work
on newer ARMv7-A systems, or even 64/32-bit Armv8-A systems.

The `armeb` architecture component specifies an Armv6 processor running in Big
Endian mode (`eb` is for big-endian - the letters are backwards because
engineers used to little-endian systems perceive big-endian numbers to be
written into memory backwards, and they thought it was funny like that).
Most Arm processors can operate in either little-endian or big-endian mode and
little-endian mode is by far the most common. However, if for whatever reason
you wish to store your Most Significant Bytes first, these targets are
available. They just aren't terribly well tested, or compatible with most
existing pre-compiled Arm libraries.

Targets that start with `armv4t` are for processors implementing the Armv4T
architecture from 1994. These include the ARM7TDMI, as found in the Nokia 6110
brick-phone and the Game Boy Advance. The 'T' stands for *Thumb* and indicate
that the processors can execute smaller 16-bit versions of some of the 32-bit
Arm instructions. This is because a Thumb is like a small version of an Arm.

Targets that start with `armv5te` are for processors implementing the Armv5TE
architecture. These are mostly from the ARM9 family, like the ARM946E-S found
in the Nintendo DS. If you are programming an Arm machine from the early
2000s, this might be what you need.

The `armv7` is arguably a misnomer, and it should be `armv7a`. This is because
it corresponds to the Application profile of Armv7 (i.e. Armv7-A), as opposed
to the Real-Time or Microcontroller profile. Processors implementing this
architecture include the Cortex-A7 and Cortex-A8.

The `thumbv7neon` component indicates support for a processor that implements
ARMv7-A (the same as `armv7`), it generates Thumb instructions (technically
Thumb-2, also known as the T32 ISA) as opposed to Arm instructions (also known
as the A32 ISA). These instructions are smaller, giving more code per KB of
RAM, but may have a performance penalty if they take two instructions to do
something Arm instructions could do in one. It's a complex trade-off and you
should be doing benchmarks to work out which is better for you, if you
strongly care about code size and/or performance. This component also enables
support for Arm's SIMD extensions, known as Neon. These extensions will
improve performance for certain kinds of repetitive operations.

## ABI Component

* `gnueabi`
* `gnueabihf`
* `musleabi`
* `musleabihf`
* `ohos`
* `uclibceabi`
* `uclibceabihf`

You will need to select the appropriate ABI to match the system you want to be
running this code on. For example, running `eabihf` code on an `eabi` system
will not work correctly.

The `gnueabi` ABI component indicates support for using the GNU C Library
(glibc), and the Arm Embedded ABI (EABI). The EABI is a replacement for the
original ABI (now called the Old ABI or OABI), and it is the standard ABI for
32-bit Arm systems. With this ABI, function parameters that are `f32` or `f64`
are passed as if they were integers, instead of being passed in FPU
registers. Generally, these targets also disable the use of the FPU entirely,
although that isn't always true.

The `gnueabihf` ABI component is like `gnueabi`, except that it supports the
'hard-float' of the EABI. That is, function parameters that are `f32` or `f64`
are passed in FPU registers. Naturally, this makes the FPU mandatory.

Most 'desktop' Linux distributions (Debian, Ubuntu, Fedora, etc) use the GNU C
Library and so you should probably select either `gnueabi` or `gnueabihf`,
depending on whether your distribution is using 'soft-float' (EABI) or
'hard-float' (EABIHF). Debian happens to offer
[both](https://wiki.debian.org/ArmEabiPort)
[kinds](https://wiki.debian.org/ArmHardFloatPort).

The `musleabi` and `musleabihf` ABI components offer support for the [musl C
library](https://musl.libc.org/). This C library can be used to create 'static
binaries' that have no run-time library requirements (a feature that glibc
does not support). There are soft-float (`eabi`) and hard-float (`eabihf`)
variants, as per the `gnu*` targets above.

The `uclibceabi` and `uclibceabihf` ABI components are for the [uClibc-ng C
library](https://uclibc-ng.org/). This is sometimes used in light-weight
embedded Linux distributions, like those created with
[buildroot](https://www.buildroot.org/).

## Cross Compilation

Unfortunately, 32-bit Arm machines are generally not the fastest around, and
they don't have much RAM. This means you are likely to be cross-compiling.

To do this, you need to give Rust a suitable linker to use - one that knows
the Arm architecture, and more importantly, knows where to find a suitable C
Library to link against.

To do that, you can add the `linker` property to your `.cargo/config.toml`.
Typically, you would refer to a suitable copy of GCC that was built as a
cross-compiler, alongside a C library.

```toml
[target.arm-unknown-linux-gnueabi]
linker = "arm-linux-gnueabi-gcc"
```

On Debian, you could install such a cross-compilation toolchain with
`apt install gcc-arm-linux-gnueabi`. For more exotic combinations, you might
need to build a bespoke version of GCC using [crosstool-ng].

[crosstool-ng]: https://github.com/crosstool-ng/crosstool-ng

Note that for GCC, all 32-bit Arm architectures are handled in the same build
- there are no separate Armv4T or Armv6 builds of GCC. The architecture is
selected with flags, like `-march=armv6`, but they aren't required for the
linker.

Let's assume we are on some Debian machine, and we want to build a basic Arm
Linux binary for a distribution using the GNU C Library, targeting Armv6 with
a hard-float ABI. Such a binary should work on a Raspberry Pi, for example.
The commands are:

```bash
sudo apt install -y gcc-arm-linux-gnueabihf
rustup target add arm-unknown-linux-gnueabihf
cargo new --bin armdemo
cd armdemo
mkdir .cargo
cat > .cargo/config.toml << EOF
[target.arm-unknown-linux-gnueabihf]
linker = "arm-linux-gnueabihf-gcc"
EOF
cargo build --target=arm-unknown-linux-gnueabihf
```

This will give us our ARM Linux binary for the GNU C Library with a soft-float ABI:

```console
$ file ./target/arm-unknown-linux-gnueabi/debug/armdemo
./target/arm-unknown-linux-gnueabi/debug/armdemo: ELF 32-bit LSB pie
  executable, ARM, EABI5 version 1 (SYSV), dynamically linked, interpreter
  /lib/ld-linux.so.3, BuildID[sha1]=dd0b9aa5ae876330fd4e2fcf393850f083ec7fcd,
  for GNU/Linux 3.2.0, with debug_info, not stripped
```

If you are building C code as part of your Rust project, you may want to
direct `cc-rs` to use an appropriate cross-compiler with the `CROSS_COMPILE`
environment variable. You may also want to set the CFLAGS environment variable
for the target. For example:

```bash
export CROSS_COMPILE=arm-linux-gnueabi
export CFLAGS_arm_unknown_linux_gnueabi="-march=armv6"
```

(Note that the dashes (`-`) turn to underscores (`_`) to form the name of the
CFLAGS environment variable)

If you are building for a Tier 3 target using `-Zbuild-std` (on Nightly Rust),
you need to set these variables as well:

```bash
export CXX_arm_unknown_linux_gnueabi=arm-linux-gnueabi-g++
export CC_arm_unknown_linux_gnueabi=arm-linux-gnueabi-gcc
cargo +nightly build -Zbuild-std --target=arm-unknown-linux-gnueabi
```
