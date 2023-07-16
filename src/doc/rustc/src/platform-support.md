# Platform Support

<style type="text/css">
    td code {
        white-space: nowrap;
    }
</style>

Support for different platforms ("targets") are organized into three tiers,
each with a different set of guarantees. For more information on the policies
for targets at each tier, see the [Target Tier Policy](target-tier-policy.md).

Targets are identified by their "target triple" which is the string to inform
the compiler what kind of output should be produced.

Component availability is tracked [here](https://rust-lang.github.io/rustup-components-history/).

## Tier 1 with Host Tools

Tier 1 targets can be thought of as "guaranteed to work". The Rust project
builds official binary releases for each tier 1 target, and automated testing
ensures that each tier 1 target builds and passes tests after each change.

Tier 1 targets with host tools additionally support running tools like `rustc`
and `cargo` natively on the target, and automated testing ensures that tests
pass for the host tools as well. This allows the target to be used as a
development platform, not just a compilation target. For the full requirements,
see [Tier 1 with Host Tools](target-tier-policy.md#tier-1-with-host-tools) in
the Target Tier Policy.

All tier 1 targets with host tools support the full standard library.

target | notes
-------|-------
`aarch64-unknown-linux-gnu` | ARM64 Linux (kernel 4.1, glibc 2.17+) [^missing-stack-probes]
`i686-pc-windows-gnu` | 32-bit MinGW (Windows 7+) [^windows-support]
`i686-pc-windows-msvc` | 32-bit MSVC (Windows 7+) [^windows-support]
`i686-unknown-linux-gnu` | 32-bit Linux (kernel 3.2+, glibc 2.17+)
`x86_64-apple-darwin` | 64-bit macOS (10.7+, Lion+)
`x86_64-pc-windows-gnu` | 64-bit MinGW (Windows 7+) [^windows-support]
`x86_64-pc-windows-msvc` | 64-bit MSVC (Windows 7+) [^windows-support]
`x86_64-unknown-linux-gnu` | 64-bit Linux (kernel 3.2+, glibc 2.17+)

[^missing-stack-probes]: Stack probes support is missing on
  `aarch64-unknown-linux-gnu`, but it's planned to be implemented in the near
  future. The implementation is tracked on [issue #77071][77071].

[^windows-support]: Only Windows 10 currently undergoes automated testing. Earlier versions of Windows rely on testing and support from the community.

[77071]: https://github.com/rust-lang/rust/issues/77071

## Tier 1

Tier 1 targets can be thought of as "guaranteed to work". The Rust project
builds official binary releases for each tier 1 target, and automated testing
ensures that each tier 1 target builds and passes tests after each change. For
the full requirements, see [Tier 1 target
policy](target-tier-policy.md#tier-1-target-policy) in the Target Tier Policy.

At this time, all Tier 1 targets are [Tier 1 with Host
Tools](#tier-1-with-host-tools).

## Tier 2 with Host Tools

Tier 2 targets can be thought of as "guaranteed to build". The Rust project
builds official binary releases for each tier 2 target, and automated builds
ensure that each tier 2 target builds after each change. Automated tests are
not always run so it's not guaranteed to produce a working build, but tier 2
targets often work to quite a good degree and patches are always welcome!

Tier 2 targets with host tools additionally support running tools like `rustc`
and `cargo` natively on the target, and automated builds ensure that the host
tools build as well. This allows the target to be used as a development
platform, not just a compilation target. For the full requirements, see [Tier 2
with Host Tools](target-tier-policy.md#tier-2-with-host-tools) in the Target
Tier Policy.

All tier 2 targets with host tools support the full standard library.

**NOTE:** The `rust-docs` component is not usually built for tier 2 targets,
so Rustup may install the documentation for a similar tier 1 target instead.

target | notes
-------|-------
`aarch64-apple-darwin` | ARM64 macOS (11.0+, Big Sur+)
`aarch64-pc-windows-msvc` | ARM64 Windows MSVC
`aarch64-unknown-linux-musl` | ARM64 Linux with MUSL
`arm-unknown-linux-gnueabi` | ARMv6 Linux (kernel 3.2, glibc 2.17)
`arm-unknown-linux-gnueabihf` | ARMv6 Linux, hardfloat (kernel 3.2, glibc 2.17)
`armv7-unknown-linux-gnueabihf` | ARMv7-A Linux, hardfloat (kernel 3.2, glibc 2.17)
[`loongarch64-unknown-linux-gnu`](platform-support/loongarch-linux.md) | LoongArch64 Linux, LP64D ABI (kernel 5.19, glibc 2.36)
`mips-unknown-linux-gnu` | MIPS Linux (kernel 4.4, glibc 2.23)
`mips64-unknown-linux-gnuabi64` | MIPS64 Linux, n64 ABI (kernel 4.4, glibc 2.23)
`mips64el-unknown-linux-gnuabi64` | MIPS64 (LE) Linux, n64 ABI (kernel 4.4, glibc 2.23)
`mipsel-unknown-linux-gnu` | MIPS (LE) Linux (kernel 4.4, glibc 2.23)
`powerpc-unknown-linux-gnu` | PowerPC Linux (kernel 3.2, glibc 2.17)
`powerpc64-unknown-linux-gnu` | PPC64 Linux (kernel 3.2, glibc 2.17)
`powerpc64le-unknown-linux-gnu` | PPC64LE Linux (kernel 3.10, glibc 2.17)
`riscv64gc-unknown-linux-gnu` | RISC-V Linux (kernel 4.20, glibc 2.29)
`s390x-unknown-linux-gnu` | S390x Linux (kernel 3.2, glibc 2.17)
`x86_64-unknown-freebsd` | 64-bit FreeBSD
`x86_64-unknown-illumos` | illumos
`x86_64-unknown-linux-musl` | 64-bit Linux with MUSL
`x86_64-unknown-netbsd` | NetBSD/amd64

## Tier 2

Tier 2 targets can be thought of as "guaranteed to build". The Rust project
builds official binary releases for each tier 2 target, and automated builds
ensure that each tier 2 target builds after each change. Automated tests are
not always run so it's not guaranteed to produce a working build, but tier 2
targets often work to quite a good degree and patches are always welcome! For
the full requirements, see [Tier 2 target
policy](target-tier-policy.md#tier-2-target-policy) in the Target Tier Policy.

The `std` column in the table below has the following meanings:

* ✓ indicates the full standard library is available.
* \* indicates the target only supports [`no_std`] development.

[`no_std`]: https://rust-embedded.github.io/book/intro/no-std.html

**NOTE:** The `rust-docs` component is not usually built for tier 2 targets,
so Rustup may install the documentation for a similar tier 1 target instead.

target | std | notes
-------|:---:|-------
`aarch64-apple-ios` | ✓ | ARM64 iOS
[`aarch64-apple-ios-sim`](platform-support/aarch64-apple-ios-sim.md) | ✓ | Apple iOS Simulator on ARM64
`aarch64-fuchsia` | ✓ | Alias for `aarch64-unknown-fuchsia`
`aarch64-unknown-fuchsia` | ✓ | ARM64 Fuchsia
[`aarch64-linux-android`](platform-support/android.md) | ✓ | ARM64 Android
`aarch64-unknown-none-softfloat` | * | Bare ARM64, softfloat
`aarch64-unknown-none` | * | Bare ARM64, hardfloat
[`aarch64-unknown-uefi`](platform-support/unknown-uefi.md) | * | ARM64 UEFI
[`arm-linux-androideabi`](platform-support/android.md) | ✓ | ARMv6 Android
`arm-unknown-linux-musleabi` | ✓ | ARMv6 Linux with MUSL
`arm-unknown-linux-musleabihf` | ✓ | ARMv6 Linux with MUSL, hardfloat
`armebv7r-none-eabi` | * | Bare ARMv7-R, Big Endian
`armebv7r-none-eabihf` | * | Bare ARMv7-R, Big Endian, hardfloat
`armv5te-unknown-linux-gnueabi` | ✓ | ARMv5TE Linux (kernel 4.4, glibc 2.23)
`armv5te-unknown-linux-musleabi` | ✓ | ARMv5TE Linux with MUSL
[`armv7-linux-androideabi`](platform-support/android.md) | ✓ | ARMv7-A Android
`armv7-unknown-linux-gnueabi` | ✓ | ARMv7-A Linux (kernel 4.15, glibc 2.27)
`armv7-unknown-linux-musleabi` | ✓ | ARMv7-A Linux with MUSL
`armv7-unknown-linux-musleabihf` | ✓ | ARMv7-A Linux with MUSL, hardfloat
`armv7a-none-eabi` | * | Bare ARMv7-A
`armv7r-none-eabi` | * | Bare ARMv7-R
`armv7r-none-eabihf` | * | Bare ARMv7-R, hardfloat
`asmjs-unknown-emscripten` | ✓ | asm.js via Emscripten
`i586-pc-windows-msvc` | * | 32-bit Windows w/o SSE
`i586-unknown-linux-gnu` | ✓ | 32-bit Linux w/o SSE (kernel 3.2, glibc 2.17)
`i586-unknown-linux-musl` | ✓ | 32-bit Linux w/o SSE, MUSL
[`i686-linux-android`](platform-support/android.md) | ✓ | 32-bit x86 Android
`i686-unknown-freebsd` | ✓ | 32-bit FreeBSD
`i686-unknown-linux-musl` | ✓ | 32-bit Linux with MUSL
[`i686-unknown-uefi`](platform-support/unknown-uefi.md) | * | 32-bit UEFI
`mips-unknown-linux-musl` | ✓ | MIPS Linux with MUSL
`mips64-unknown-linux-muslabi64` | ✓ | MIPS64 Linux, n64 ABI, MUSL
`mips64el-unknown-linux-muslabi64` | ✓ | MIPS64 (LE) Linux, n64 ABI, MUSL
`mipsel-unknown-linux-musl` | ✓ | MIPS (LE) Linux with MUSL
`nvptx64-nvidia-cuda` | * | --emit=asm generates PTX code that [runs on NVIDIA GPUs]
`riscv32i-unknown-none-elf` | * | Bare RISC-V (RV32I ISA)
`riscv32imac-unknown-none-elf` | * | Bare RISC-V (RV32IMAC ISA)
`riscv32imc-unknown-none-elf` | * | Bare RISC-V (RV32IMC ISA)
`riscv64gc-unknown-none-elf` | * | Bare RISC-V (RV64IMAFDC ISA)
`riscv64imac-unknown-none-elf` | * | Bare RISC-V (RV64IMAC ISA)
`sparc64-unknown-linux-gnu` | ✓ | SPARC Linux (kernel 4.4, glibc 2.23)
`sparcv9-sun-solaris` | ✓ | SPARC Solaris 10/11, illumos
`thumbv6m-none-eabi` | * | Bare ARMv6-M
`thumbv7em-none-eabi` | * | Bare ARMv7E-M
`thumbv7em-none-eabihf` | * | Bare ARMV7E-M, hardfloat
`thumbv7m-none-eabi` | * | Bare ARMv7-M
[`thumbv7neon-linux-androideabi`](platform-support/android.md) | ✓ | Thumb2-mode ARMv7-A Android with NEON
`thumbv7neon-unknown-linux-gnueabihf` | ✓ | Thumb2-mode ARMv7-A Linux with NEON (kernel 4.4, glibc 2.23)
`thumbv8m.base-none-eabi` | * | Bare ARMv8-M Baseline
`thumbv8m.main-none-eabi` | * | Bare ARMv8-M Mainline
`thumbv8m.main-none-eabihf` | * | Bare ARMv8-M Mainline, hardfloat
`wasm32-unknown-emscripten` | ✓ | WebAssembly via Emscripten
`wasm32-unknown-unknown` | ✓ | WebAssembly
`wasm32-wasi` | ✓ | WebAssembly with WASI
`x86_64-apple-ios` | ✓ | 64-bit x86 iOS
[`x86_64-fortanix-unknown-sgx`](platform-support/x86_64-fortanix-unknown-sgx.md) | ✓ | [Fortanix ABI] for 64-bit Intel SGX
`x86_64-fuchsia` | ✓ | Alias for `x86_64-unknown-fuchsia`
`x86_64-unknown-fuchsia` | ✓ | 64-bit Fuchsia
[`x86_64-linux-android`](platform-support/android.md) | ✓ | 64-bit x86 Android
`x86_64-pc-solaris` | ✓ | 64-bit Solaris 10/11, illumos
`x86_64-unknown-linux-gnux32` | ✓ | 64-bit Linux (x32 ABI) (kernel 4.15, glibc 2.27)
[`x86_64-unknown-none`](platform-support/x86_64-unknown-none.md) | * | Freestanding/bare-metal x86_64, softfloat
`x86_64-unknown-redox` | ✓ | Redox OS
[`x86_64-unknown-uefi`](platform-support/unknown-uefi.md) | * | 64-bit UEFI

[Fortanix ABI]: https://edp.fortanix.com/

## Tier 3

Tier 3 targets are those which the Rust codebase has support for, but which the
Rust project does not build or test automatically, so they may or may not work.
Official builds are not available. For the full requirements, see [Tier 3
target policy](target-tier-policy.md#tier-3-target-policy) in the Target Tier
Policy.

The `std` column in the table below has the following meanings:

* ✓ indicates the full standard library is available.
* \* indicates the target only supports [`no_std`] development.
* ? indicates the standard library support is unknown or a work-in-progress.

[`no_std`]: https://rust-embedded.github.io/book/intro/no-std.html

The `host` column indicates whether the codebase includes support for building
host tools.

target | std | host | notes
-------|:---:|:----:|-------
`aarch64-apple-ios-macabi` | ? |  | Apple Catalyst on ARM64
[`aarch64-apple-tvos`](platform-support/apple-tvos.md) | ? |  | ARM64 tvOS
[`aarch64-apple-watchos-sim`](platform-support/apple-watchos.md) | ✓ |  | ARM64 Apple WatchOS Simulator
[`aarch64-kmc-solid_asp3`](platform-support/kmc-solid.md) | ✓ |  | ARM64 SOLID with TOPPERS/ASP3
[`aarch64-nintendo-switch-freestanding`](platform-support/aarch64-nintendo-switch-freestanding.md) | * |  | ARM64 Nintendo Switch, Horizon
[`aarch64-pc-windows-gnullvm`](platform-support/pc-windows-gnullvm.md) | ✓ | ✓ |
[`aarch64-unknown-linux-ohos`](platform-support/openharmony.md) | ✓ |  | ARM64 OpenHarmony |
[`aarch64-unknown-nto-qnx710`](platform-support/nto-qnx.md) | ✓ |  | ARM64 QNX Neutrino 7.1 RTOS |
`aarch64-unknown-freebsd` | ✓ | ✓ | ARM64 FreeBSD
`aarch64-unknown-hermit` | ✓ |  | ARM64 HermitCore
`aarch64-unknown-linux-gnu_ilp32` | ✓ | ✓ | ARM64 Linux (ILP32 ABI)
[`aarch64-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | ARM64 NetBSD
[`aarch64-unknown-openbsd`](platform-support/openbsd.md) | ✓ | ✓ | ARM64 OpenBSD
`aarch64-unknown-redox` | ? |  | ARM64 Redox OS
`aarch64-uwp-windows-msvc` | ? |  |
`aarch64-wrs-vxworks` | ? |  |
`aarch64_be-unknown-linux-gnu_ilp32` | ✓ | ✓ | ARM64 Linux (big-endian, ILP32 ABI)
`aarch64_be-unknown-linux-gnu` | ✓ | ✓ | ARM64 Linux (big-endian)
[`aarch64_be-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | ARM64 NetBSD (big-endian)
[`arm64_32-apple-watchos`](platform-support/apple-watchos.md) | ✓ | | ARM Apple WatchOS 64-bit with 32-bit pointers
[`armeb-unknown-linux-gnueabi`](platform-support/armeb-unknown-linux-gnueabi.md) | ✓ | ? | ARM BE8 the default ARM big-endian architecture since [ARMv6](https://developer.arm.com/documentation/101754/0616/armlink-Reference/armlink-Command-line-Options/--be8?lang=en).
`armv4t-none-eabi` | * |  | Bare ARMv4T
`armv4t-unknown-linux-gnueabi` | ? |  | ARMv4T Linux
[`armv5te-none-eabi`](platform-support/armv5te-none-eabi.md) | * | | Bare ARMv5TE
`armv5te-unknown-linux-uclibceabi` | ? |  | ARMv5TE Linux with uClibc
`armv6-unknown-freebsd` | ✓ | ✓ | ARMv6 FreeBSD
[`armv6-unknown-netbsd-eabihf`](platform-support/netbsd.md) | ✓ | ✓ | ARMv6 NetBSD w/hard-float
[`armv6k-nintendo-3ds`](platform-support/armv6k-nintendo-3ds.md) | ? |  | ARMv6K Nintendo 3DS, Horizon (Requires devkitARM toolchain)
`armv7-apple-ios` | ✓ |  | ARMv7-A Cortex-A8 iOS
[`armv7-sony-vita-newlibeabihf`](platform-support/armv7-sony-vita-newlibeabihf.md) | ? |  | ARMv7-A Cortex-A9 Sony PlayStation Vita (requires VITASDK toolchain)
[`armv7-unknown-linux-ohos`](platform-support/openharmony.md) | ✓ |  | ARMv7-A OpenHarmony |
[`armv7-unknown-linux-uclibceabi`](platform-support/armv7-unknown-linux-uclibceabi.md) | ✓ | ✓ | ARMv7-A Linux with uClibc, softfloat
[`armv7-unknown-linux-uclibceabihf`](platform-support/armv7-unknown-linux-uclibceabihf.md) | ✓ | ? | ARMv7-A Linux with uClibc, hardfloat
`armv7-unknown-freebsd` | ✓ | ✓ | ARMv7-A FreeBSD
[`armv7-unknown-netbsd-eabihf`](platform-support/netbsd.md) | ✓ | ✓ | ARMv7-A NetBSD w/hard-float
`armv7-wrs-vxworks-eabihf` | ? |  | ARMv7-A for VxWorks
[`armv7a-kmc-solid_asp3-eabi`](platform-support/kmc-solid.md) | ✓ |  | ARM SOLID with TOPPERS/ASP3
[`armv7a-kmc-solid_asp3-eabihf`](platform-support/kmc-solid.md) | ✓ |  | ARM SOLID with TOPPERS/ASP3, hardfloat
`armv7a-none-eabihf` | * | | Bare ARMv7-A, hardfloat
[`armv7k-apple-watchos`](platform-support/apple-watchos.md) | ✓ | | ARMv7-A Apple WatchOS
`armv7s-apple-ios` | ✓ |  | ARMv7-A Apple-A6 Apple iOS
`avr-unknown-gnu-atmega328` | * |  | AVR. Requires `-Z build-std=core`
`bpfeb-unknown-none` | * |  | BPF (big endian)
`bpfel-unknown-none` | * |  | BPF (little endian)
`hexagon-unknown-linux-musl` | ? |  |
`i386-apple-ios` | ✓ |  | 32-bit x86 iOS
[`i586-pc-nto-qnx700`](platform-support/nto-qnx.md) | * |  | 32-bit x86 QNX Neutrino 7.0 RTOS |
`i686-apple-darwin` | ✓ | ✓ | 32-bit macOS (10.7+, Lion+)
`i686-pc-windows-msvc` | * |  | 32-bit Windows XP support
`i686-unknown-haiku` | ✓ | ✓ | 32-bit Haiku
[`i686-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | NetBSD/i386 with SSE2
[`i686-unknown-openbsd`](platform-support/openbsd.md) | ✓ | ✓ | 32-bit OpenBSD
`i686-uwp-windows-gnu` | ? |  |
`i686-uwp-windows-msvc` | ? |  |
`i686-wrs-vxworks` | ? |  |
[`loongarch64-unknown-none`](platform-support/loongarch-none.md) | * | LoongArch64 Bare-metal (LP64D ABI)
[`loongarch64-unknown-none-softfloat`](platform-support/loongarch-none.md) | * | LoongArch64 Bare-metal (LP64S ABI)
[`m68k-unknown-linux-gnu`](platform-support/m68k-unknown-linux-gnu.md) | ? |  | Motorola 680x0 Linux
`mips-unknown-linux-uclibc` | ✓ |  | MIPS Linux with uClibc
[`mips64-openwrt-linux-musl`](platform-support/mips64-openwrt-linux-musl.md) | ? |  | MIPS64 for OpenWrt Linux MUSL
`mipsel-sony-psp` | * |  | MIPS (LE) Sony PlayStation Portable (PSP)
[`mipsel-sony-psx`](platform-support/mipsel-sony-psx.md) | * |  | MIPS (LE) Sony PlayStation 1 (PSX)
`mipsel-unknown-linux-uclibc` | ✓ |  | MIPS (LE) Linux with uClibc
`mipsel-unknown-none` | * |  | Bare MIPS (LE) softfloat
[`mipsisa32r6-unknown-linux-gnu`](platform-support/mips-release-6.md) | ? |  | 32-bit MIPS Release 6 Big Endian
[`mipsisa32r6el-unknown-linux-gnu`](platform-support/mips-release-6.md) | ? |  | 32-bit MIPS Release 6 Little Endian
[`mipsisa64r6-unknown-linux-gnuabi64`](platform-support/mips-release-6.md) | ? |  | 64-bit MIPS Release 6 Big Endian
[`mipsisa64r6el-unknown-linux-gnuabi64`](platform-support/mips-release-6.md) | ✓ | ✓ | 64-bit MIPS Release 6 Little Endian
`msp430-none-elf` | * |  | 16-bit MSP430 microcontrollers
`powerpc-unknown-linux-gnuspe` | ✓ |  | PowerPC SPE Linux
`powerpc-unknown-linux-musl` | ? |  |
[`powerpc-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | NetBSD 32-bit powerpc systems
`powerpc-unknown-openbsd` | ? |  |
`powerpc-wrs-vxworks-spe` | ? |  |
`powerpc-wrs-vxworks` | ? |  |
`powerpc64-unknown-freebsd` | ✓ | ✓ | PPC64 FreeBSD (ELFv1 and ELFv2)
`powerpc64le-unknown-freebsd` |   |   | PPC64LE FreeBSD
`powerpc-unknown-freebsd` |   |   | PowerPC FreeBSD
`powerpc64-unknown-linux-musl` | ? |  |
`powerpc64-wrs-vxworks` | ? |  |
`powerpc64le-unknown-linux-musl` | ? |  |
[`powerpc64-unknown-openbsd`](platform-support/openbsd.md) | ✓ | ✓ | OpenBSD/powerpc64
`powerpc64-ibm-aix` | ? |  | 64-bit AIX (7.2 and newer)
`riscv32gc-unknown-linux-gnu` |   |   | RISC-V Linux (kernel 5.4, glibc 2.33)
`riscv32gc-unknown-linux-musl` |   |   | RISC-V Linux (kernel 5.4, musl + RISCV32 support patches)
`riscv32im-unknown-none-elf` | * |  | Bare RISC-V (RV32IM ISA)
[`riscv32imac-unknown-xous-elf`](platform-support/riscv32imac-unknown-xous-elf.md) | ? |  | RISC-V Xous (RV32IMAC ISA)
[`riscv32imc-esp-espidf`](platform-support/esp-idf.md) | ✓ |  | RISC-V ESP-IDF
[`riscv32imac-esp-espidf`](platform-support/esp-idf.md) | ✓ |  | RISC-V ESP-IDF
`riscv64gc-unknown-freebsd` |   |   | RISC-V FreeBSD
`riscv64gc-unknown-fuchsia` |   |   | RISC-V Fuchsia
`riscv64gc-unknown-linux-musl` |   |   | RISC-V Linux (kernel 4.20, musl 1.2.0)
[`riscv64gc-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | RISC-V NetBSD
[`riscv64gc-unknown-openbsd`](platform-support/openbsd.md) | ✓ | ✓ | OpenBSD/riscv64
`s390x-unknown-linux-musl` |  |  | S390x Linux (kernel 3.2, MUSL)
`sparc-unknown-linux-gnu` | ✓ |  | 32-bit SPARC Linux
[`sparc64-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | NetBSD/sparc64
[`sparc64-unknown-openbsd`](platform-support/openbsd.md) | ✓ | ✓ | OpenBSD/sparc64
`thumbv4t-none-eabi` | * |  | Thumb-mode Bare ARMv4T
[`thumbv5te-none-eabi`](platform-support/armv5te-none-eabi.md) | * | | Thumb-mode Bare ARMv5TE
`thumbv7a-pc-windows-msvc` | ? |  |
`thumbv7a-uwp-windows-msvc` | ✓ |  |
`thumbv7neon-unknown-linux-musleabihf` | ? |  | Thumb2-mode ARMv7-A Linux with NEON, MUSL
[`wasm64-unknown-unknown`](platform-support/wasm64-unknown-unknown.md) | ? |  | WebAssembly
`x86_64-apple-ios-macabi` | ✓ |  | Apple Catalyst on x86_64
[`x86_64-apple-tvos`](platform-support/apple-tvos.md) | ? | | x86 64-bit tvOS
[`x86_64-apple-watchos-sim`](platform-support/apple-watchos.md) | ✓ | | x86 64-bit Apple WatchOS simulator
[`x86_64-pc-nto-qnx710`](platform-support/nto-qnx.md) | ✓ |  | x86 64-bit QNX Neutrino 7.1 RTOS |
[`x86_64-pc-windows-gnullvm`](platform-support/pc-windows-gnullvm.md) | ✓ | ✓ |
`x86_64-pc-windows-msvc` | * |  | 64-bit Windows XP support
`x86_64-sun-solaris` | ? |  | Deprecated target for 64-bit Solaris 10/11, illumos
`x86_64-unknown-dragonfly` | ✓ | ✓ | 64-bit DragonFlyBSD
`x86_64-unknown-haiku` | ✓ | ✓ | 64-bit Haiku
`x86_64-unknown-hermit` | ✓ |  | HermitCore
`x86_64-unknown-l4re-uclibc` | ? |  |
[`x86_64-unknown-openbsd`](platform-support/openbsd.md) | ✓ | ✓ | 64-bit OpenBSD
`x86_64-uwp-windows-gnu` | ✓ |  |
`x86_64-uwp-windows-msvc` | ✓ |  |
`x86_64-wrs-vxworks` | ? |  |
`x86_64h-apple-darwin` | ✓ | ✓ | macOS with late-gen Intel (at least Haswell)

[runs on NVIDIA GPUs]: https://github.com/japaric-archived/nvptx#targets
