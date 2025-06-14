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
[`aarch64-apple-darwin`](platform-support/apple-darwin.md) | ARM64 macOS (11.0+, Big Sur+)
`aarch64-unknown-linux-gnu` | ARM64 Linux (kernel 4.1+, glibc 2.17+)
[`i686-pc-windows-msvc`](platform-support/windows-msvc.md) | 32-bit MSVC (Windows 10+, Windows Server 2016+, Pentium 4) [^x86_32-floats-return-ABI] [^win32-msvc-alignment]
`i686-unknown-linux-gnu` | 32-bit Linux (kernel 3.2+, glibc 2.17+, Pentium 4) [^x86_32-floats-return-ABI]
[`x86_64-apple-darwin`](platform-support/apple-darwin.md) | 64-bit macOS (10.12+, Sierra+)
[`x86_64-pc-windows-gnu`](platform-support/windows-gnu.md) | 64-bit MinGW (Windows 10+, Windows Server 2016+)
[`x86_64-pc-windows-msvc`](platform-support/windows-msvc.md) | 64-bit MSVC (Windows 10+, Windows Server 2016+)
`x86_64-unknown-linux-gnu` | 64-bit Linux (kernel 3.2+, glibc 2.17+)

[^x86_32-floats-return-ABI]: Due to limitations of the C ABI, floating-point support on `i686` targets is non-compliant: floating-point return values are passed via an x87 register, so NaN payload bits can be lost. Functions with the default Rust ABI are not affected. See [issue #115567][x86-32-float-return-issue].

[^win32-msvc-alignment]: Due to non-standard behavior of MSVC, native C code on this target can cause types with an alignment of more than 4 bytes to be incorrectly aligned to only 4 bytes (this affects, e.g., `u64` and `i64`). Rust applies some mitigations to reduce the impact of this issue, but this can still cause unsoundness due to unsafe code that (correctly) assumes that references are always properly aligned. See [issue #112480](https://github.com/rust-lang/rust/issues/112480).

[77071]: https://github.com/rust-lang/rust/issues/77071
[x86-32-float-return-issue]: https://github.com/rust-lang/rust/issues/115567

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
builds official binary releases of the standard library (or, in some cases,
only the `core` library) for each tier 2 target, and automated builds
ensure that each tier 2 target can be used as build target after each change. Automated tests are
not always run so it's not guaranteed to produce a working build, but tier 2
targets often work to quite a good degree and patches are always welcome!

Tier 2 target-specific code is not closely scrutinized by Rust team(s) when
modifications are made. Bugs are possible in all code, but the level of quality
control for these targets is likely to be lower. See [library team
policy](https://std-dev-guide.rust-lang.org/policy/target-code.html) for
details on the review practices for standard library code.

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
[`aarch64-pc-windows-msvc`](platform-support/windows-msvc.md) | ARM64 Windows MSVC
`aarch64-unknown-linux-musl` | ARM64 Linux with musl 1.2.3
[`aarch64-unknown-linux-ohos`](platform-support/openharmony.md) | ARM64 OpenHarmony
`arm-unknown-linux-gnueabi` | Armv6 Linux (kernel 3.2+, glibc 2.17)
`arm-unknown-linux-gnueabihf` | Armv6 Linux, hardfloat (kernel 3.2+, glibc 2.17)
`armv7-unknown-linux-gnueabihf` | Armv7-A Linux, hardfloat (kernel 3.2+, glibc 2.17)
[`armv7-unknown-linux-ohos`](platform-support/openharmony.md) | Armv7-A OpenHarmony
[`loongarch64-unknown-linux-gnu`](platform-support/loongarch-linux.md) | LoongArch64 Linux, LP64D ABI (kernel 5.19+, glibc 2.36)
[`loongarch64-unknown-linux-musl`](platform-support/loongarch-linux.md) | LoongArch64 Linux, LP64D ABI (kernel 5.19+, musl 1.2.5)
[`i686-pc-windows-gnu`](platform-support/windows-gnu.md) | 32-bit MinGW (Windows 10+, Windows Server 2016+, Pentium 4) [^x86_32-floats-return-ABI] [^win32-msvc-alignment]
`powerpc-unknown-linux-gnu` | PowerPC Linux (kernel 3.2+, glibc 2.17)
`powerpc64-unknown-linux-gnu` | PPC64 Linux (kernel 3.2+, glibc 2.17)
[`powerpc64le-unknown-linux-gnu`](platform-support/powerpc64le-unknown-linux-gnu.md) | PPC64LE Linux (kernel 3.10+, glibc 2.17)
[`powerpc64le-unknown-linux-musl`](platform-support/powerpc64le-unknown-linux-musl.md) | PPC64LE Linux (kernel 4.19+, musl 1.2.3)
[`riscv64gc-unknown-linux-gnu`](platform-support/riscv64gc-unknown-linux-gnu.md) | RISC-V Linux (kernel 4.20+, glibc 2.29)
[`riscv64gc-unknown-linux-musl`](platform-support/riscv64gc-unknown-linux-musl.md) | RISC-V Linux (kernel 4.20+, musl 1.2.3)
[`s390x-unknown-linux-gnu`](platform-support/s390x-unknown-linux-gnu.md) | S390x Linux (kernel 3.2+, glibc 2.17)
[`x86_64-unknown-freebsd`](platform-support/freebsd.md) | 64-bit x86 FreeBSD
[`x86_64-unknown-illumos`](platform-support/illumos.md) | illumos
`x86_64-unknown-linux-musl` | 64-bit Linux with musl 1.2.3
[`x86_64-unknown-linux-ohos`](platform-support/openharmony.md) | x86_64 OpenHarmony
[`x86_64-unknown-netbsd`](platform-support/netbsd.md) | NetBSD/amd64
[`x86_64-pc-solaris`](platform-support/solaris.md) | 64-bit x86 Solaris 11.4
[`sparcv9-sun-solaris`](platform-support/solaris.md) | SPARC V9 Solaris 11.4

## Tier 2 without Host Tools

Tier 2 targets can be thought of as "guaranteed to build". The Rust project
builds official binary releases of the standard library (or, in some cases,
only the `core` library) for each tier 2 target, and automated builds
ensure that each tier 2 target can be used as build target after each change. Automated tests are
not always run so it's not guaranteed to produce a working build, but tier 2
targets often work to quite a good degree and patches are always welcome! For
the full requirements, see [Tier 2 target
policy](target-tier-policy.md#tier-2-target-policy) in the Target Tier Policy.

The `std` column in the table below has the following meanings:

* ✓ indicates the full standard library is available.
* \* indicates the target only supports [`no_std`] development.
* ? indicates the standard library support is a work-in-progress.

[`no_std`]: https://rust-embedded.github.io/book/intro/no-std.html

Tier 2 target-specific code is not closely scrutinized by Rust team(s) when
modifications are made. Bugs are possible in all code, but the level of quality
control for these targets is likely to be lower. See [library team
policy](https://std-dev-guide.rust-lang.org/policy/target-code.html) for
details on the review practices for standard library code.

**NOTE:** The `rust-docs` component is not usually built for tier 2 targets,
so Rustup may install the documentation for a similar tier 1 target instead.

target | std | notes
-------|:---:|-------
[`aarch64-apple-ios`](platform-support/apple-ios.md) | ✓ | ARM64 iOS
[`aarch64-apple-ios-macabi`](platform-support/apple-ios-macabi.md) | ✓ | Mac Catalyst on ARM64
[`aarch64-apple-ios-sim`](platform-support/apple-ios.md) | ✓ | Apple iOS Simulator on ARM64
[`aarch64-linux-android`](platform-support/android.md) | ✓ | ARM64 Android
[`aarch64-pc-windows-gnullvm`](platform-support/windows-gnullvm.md) | ✓ | ARM64 MinGW (Windows 10+), LLVM ABI
[`aarch64-unknown-fuchsia`](platform-support/fuchsia.md) | ✓ | ARM64 Fuchsia
`aarch64-unknown-none` | * | Bare ARM64, hardfloat
`aarch64-unknown-none-softfloat` | * | Bare ARM64, softfloat
[`aarch64-unknown-uefi`](platform-support/unknown-uefi.md) | ? | ARM64 UEFI
[`arm-linux-androideabi`](platform-support/android.md) | ✓ | Armv6 Android
`arm-unknown-linux-musleabi` | ✓ | Armv6 Linux with musl 1.2.3
`arm-unknown-linux-musleabihf` | ✓ | Armv6 Linux with musl 1.2.3, hardfloat
[`arm64ec-pc-windows-msvc`](platform-support/arm64ec-pc-windows-msvc.md) | ✓ | Arm64EC Windows MSVC
[`armebv7r-none-eabi`](platform-support/armv7r-none-eabi.md) | * | Bare Armv7-R, Big Endian
[`armebv7r-none-eabihf`](platform-support/armv7r-none-eabi.md) | * | Bare Armv7-R, Big Endian, hardfloat
[`armv5te-unknown-linux-gnueabi`](platform-support/armv5te-unknown-linux-gnueabi.md) | ✓ | Armv5TE Linux (kernel 4.4+, glibc 2.23)
`armv5te-unknown-linux-musleabi` | ✓ | Armv5TE Linux with musl 1.2.3
[`armv7-linux-androideabi`](platform-support/android.md) | ✓ | Armv7-A Android
`armv7-unknown-linux-gnueabi` | ✓ | Armv7-A Linux (kernel 4.15+, glibc 2.27)
`armv7-unknown-linux-musleabi` | ✓ | Armv7-A Linux with musl 1.2.3
`armv7-unknown-linux-musleabihf` | ✓ | Armv7-A Linux with musl 1.2.3, hardfloat
[`armv7a-none-eabi`](platform-support/arm-none-eabi.md) | * | Bare Armv7-A
[`armv7r-none-eabi`](platform-support/armv7r-none-eabi.md) | * | Bare Armv7-R
[`armv7r-none-eabihf`](platform-support/armv7r-none-eabi.md) | * | Bare Armv7-R, hardfloat
`i586-unknown-linux-gnu` | ✓ | 32-bit Linux (kernel 3.2+, glibc 2.17, original Pentium) [^x86_32-floats-x87]
`i586-unknown-linux-musl` | ✓ | 32-bit Linux (musl 1.2.3, original Pentium) [^x86_32-floats-x87]
[`i686-linux-android`](platform-support/android.md) | ✓ | 32-bit x86 Android ([Pentium 4 plus various extensions](https://developer.android.com/ndk/guides/abis.html#x86)) [^x86_32-floats-return-ABI]
[`i686-pc-windows-gnullvm`](platform-support/windows-gnullvm.md) | ✓ | 32-bit x86 MinGW (Windows 10+, Pentium 4), LLVM ABI [^x86_32-floats-return-ABI]
[`i686-unknown-freebsd`](platform-support/freebsd.md) | ✓ | 32-bit x86 FreeBSD (Pentium 4) [^x86_32-floats-return-ABI]
`i686-unknown-linux-musl` | ✓ | 32-bit Linux with musl 1.2.3 (Pentium 4) [^x86_32-floats-return-ABI]
[`i686-unknown-uefi`](platform-support/unknown-uefi.md) | ? | 32-bit UEFI (Pentium 4, softfloat) [^win32-msvc-alignment]
[`loongarch64-unknown-none`](platform-support/loongarch-none.md) | * | LoongArch64 Bare-metal (LP64D ABI)
[`loongarch64-unknown-none-softfloat`](platform-support/loongarch-none.md) | * | LoongArch64 Bare-metal (LP64S ABI)
[`nvptx64-nvidia-cuda`](platform-support/nvptx64-nvidia-cuda.md) | * | --emit=asm generates PTX code that [runs on NVIDIA GPUs]
[`riscv32i-unknown-none-elf`](platform-support/riscv32-unknown-none-elf.md) | * | Bare RISC-V (RV32I ISA)
[`riscv32im-unknown-none-elf`](platform-support/riscv32-unknown-none-elf.md) | * | Bare RISC-V (RV32IM ISA)
[`riscv32imac-unknown-none-elf`](platform-support/riscv32-unknown-none-elf.md) | * | Bare RISC-V (RV32IMAC ISA)
[`riscv32imafc-unknown-none-elf`](platform-support/riscv32-unknown-none-elf.md) | * | Bare RISC-V (RV32IMAFC ISA)
[`riscv32imc-unknown-none-elf`](platform-support/riscv32-unknown-none-elf.md) | * | Bare RISC-V (RV32IMC ISA)
`riscv64gc-unknown-none-elf` | * | Bare RISC-V (RV64IMAFDC ISA)
`riscv64imac-unknown-none-elf` | * | Bare RISC-V (RV64IMAC ISA)
`sparc64-unknown-linux-gnu` | ✓ | SPARC Linux (kernel 4.4+, glibc 2.23)
[`thumbv6m-none-eabi`](platform-support/thumbv6m-none-eabi.md) | * | Bare Armv6-M
[`thumbv7em-none-eabi`](platform-support/thumbv7em-none-eabi.md) | * | Bare Armv7E-M
[`thumbv7em-none-eabihf`](platform-support/thumbv7em-none-eabi.md) | * | Bare Armv7E-M, hardfloat
[`thumbv7m-none-eabi`](platform-support/thumbv7m-none-eabi.md) | * | Bare Armv7-M
[`thumbv7neon-linux-androideabi`](platform-support/android.md) | ✓ | Thumb2-mode Armv7-A Android with NEON
`thumbv7neon-unknown-linux-gnueabihf` | ✓ | Thumb2-mode Armv7-A Linux with NEON (kernel 4.4+, glibc 2.23)
[`thumbv8m.base-none-eabi`](platform-support/thumbv8m.base-none-eabi.md) | * | Bare Armv8-M Baseline
[`thumbv8m.main-none-eabi`](platform-support/thumbv8m.main-none-eabi.md) | * | Bare Armv8-M Mainline
[`thumbv8m.main-none-eabihf`](platform-support/thumbv8m.main-none-eabi.md) | * | Bare Armv8-M Mainline, hardfloat
[`wasm32-unknown-emscripten`](platform-support/wasm32-unknown-emscripten.md) | ✓ | WebAssembly via Emscripten
[`wasm32-unknown-unknown`](platform-support/wasm32-unknown-unknown.md) | ✓ | WebAssembly
[`wasm32-wasip1`](platform-support/wasm32-wasip1.md) | ✓ | WebAssembly with WASIp1
[`wasm32-wasip1-threads`](platform-support/wasm32-wasip1-threads.md) | ✓ | WebAssembly with WASI Preview 1 and threads
[`wasm32-wasip2`](platform-support/wasm32-wasip2.md) | ✓ | WebAssembly with WASIp2
[`wasm32v1-none`](platform-support/wasm32v1-none.md) | * | WebAssembly limited to 1.0 features and no imports
[`x86_64-apple-ios`](platform-support/apple-ios.md) | ✓ | 64-bit x86 iOS
[`x86_64-apple-ios-macabi`](platform-support/apple-ios-macabi.md) | ✓ | Mac Catalyst on x86_64
[`x86_64-fortanix-unknown-sgx`](platform-support/x86_64-fortanix-unknown-sgx.md) | ✓ | [Fortanix ABI] for 64-bit Intel SGX
[`x86_64-linux-android`](platform-support/android.md) | ✓ | 64-bit x86 Android
[`x86_64-pc-windows-gnullvm`](platform-support/windows-gnullvm.md) | ✓ | 64-bit x86 MinGW (Windows 10+), LLVM ABI
[`x86_64-unknown-fuchsia`](platform-support/fuchsia.md) | ✓ | 64-bit x86 Fuchsia
`x86_64-unknown-linux-gnux32` | ✓ | 64-bit Linux (x32 ABI) (kernel 4.15+, glibc 2.27)
[`x86_64-unknown-none`](platform-support/x86_64-unknown-none.md) | * | Freestanding/bare-metal x86_64, softfloat
[`x86_64-unknown-redox`](platform-support/redox.md) | ✓ | Redox OS
[`x86_64-unknown-uefi`](platform-support/unknown-uefi.md) | ? | 64-bit UEFI

[^x86_32-floats-x87]: Floating-point support on `i586` targets is non-compliant: the `x87` registers and instructions used for these targets do not provide IEEE-754-compliant behavior, in particular when it comes to rounding and NaN payload bits. See [issue #114479][x86-32-float-issue].

[x86-32-float-issue]: https://github.com/rust-lang/rust/issues/114479

[wasi-rename]: https://github.com/rust-lang/compiler-team/issues/607

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

Tier 3 target-specific code is not closely scrutinized by Rust team(s) when
modifications are made. Bugs are possible in all code, but the level of quality
control for these targets is likely to be lower. See [library team
policy](https://std-dev-guide.rust-lang.org/policy/target-code.html) for
details on the review practices for standard library code.

The `host` column indicates whether the codebase includes support for building
host tools.

target | std | host | notes
-------|:---:|:----:|-------
[`aarch64-apple-tvos`](platform-support/apple-tvos.md) | ✓ |  | ARM64 tvOS
[`aarch64-apple-tvos-sim`](platform-support/apple-tvos.md) | ✓ |  | ARM64 tvOS Simulator
[`aarch64-apple-visionos`](platform-support/apple-visionos.md) | ✓ |  | ARM64 Apple visionOS
[`aarch64-apple-visionos-sim`](platform-support/apple-visionos.md) | ✓ |  | ARM64 Apple visionOS Simulator
[`aarch64-apple-watchos`](platform-support/apple-watchos.md) | ✓ |  | ARM64 Apple WatchOS
[`aarch64-apple-watchos-sim`](platform-support/apple-watchos.md) | ✓ |  | ARM64 Apple WatchOS Simulator
[`aarch64-kmc-solid_asp3`](platform-support/kmc-solid.md) | ✓ |  | ARM64 SOLID with TOPPERS/ASP3
[`aarch64-nintendo-switch-freestanding`](platform-support/aarch64-nintendo-switch-freestanding.md) | * |  | ARM64 Nintendo Switch, Horizon
[`aarch64-unknown-freebsd`](platform-support/freebsd.md) | ✓ | ✓ | ARM64 FreeBSD
[`aarch64-unknown-hermit`](platform-support/hermit.md) | ✓ |  | ARM64 Hermit
[`aarch64-unknown-illumos`](platform-support/illumos.md) | ✓ | ✓ | ARM64 illumos
`aarch64-unknown-linux-gnu_ilp32` | ✓ | ✓ | ARM64 Linux (ILP32 ABI)
[`aarch64-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | ARM64 NetBSD
[`aarch64-unknown-nto-qnx700`](platform-support/nto-qnx.md) | ? |  | ARM64 QNX Neutrino 7.0 RTOS |
[`aarch64-unknown-nto-qnx710`](platform-support/nto-qnx.md) | ✓ |  | ARM64 QNX Neutrino 7.1 RTOS with default network stack (io-pkt) |
[`aarch64-unknown-nto-qnx710_iosock`](platform-support/nto-qnx.md) | ✓ |  | ARM64 QNX Neutrino 7.1 RTOS with new network stack (io-sock) |
[`aarch64-unknown-nto-qnx800`](platform-support/nto-qnx.md) | ✓ |  | ARM64 QNX Neutrino 8.0 RTOS |
[`aarch64-unknown-nuttx`](platform-support/nuttx.md) | ✓ |  | ARM64 with NuttX
[`aarch64-unknown-openbsd`](platform-support/openbsd.md) | ✓ | ✓ | ARM64 OpenBSD
[`aarch64-unknown-redox`](platform-support/redox.md) | ✓ |  | ARM64 Redox OS
[`aarch64-unknown-teeos`](platform-support/aarch64-unknown-teeos.md) | ? |  | ARM64 TEEOS |
[`aarch64-unknown-trusty`](platform-support/trusty.md) | ✓ |  |
[`aarch64-uwp-windows-msvc`](platform-support/uwp-windows-msvc.md) | ✓ |  |
[`aarch64-wrs-vxworks`](platform-support/vxworks.md) | ✓ |  | ARM64 VxWorks OS
`aarch64_be-unknown-linux-gnu` | ✓ | ✓ | ARM64 Linux (big-endian)
`aarch64_be-unknown-linux-gnu_ilp32` | ✓ | ✓ | ARM64 Linux (big-endian, ILP32 ABI)
[`aarch64_be-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | ARM64 NetBSD (big-endian)
[`amdgcn-amd-amdhsa`](platform-support/amdgcn-amd-amdhsa.md) | * |  | `-Ctarget-cpu=gfx...` to specify [the AMD GPU] to compile for
[`arm64_32-apple-watchos`](platform-support/apple-watchos.md) | ✓ |  | Arm Apple WatchOS 64-bit with 32-bit pointers
[`arm64e-apple-darwin`](platform-support/arm64e-apple-darwin.md)  | ✓ | ✓ | ARM64e Apple Darwin
[`arm64e-apple-ios`](platform-support/arm64e-apple-ios.md) | ✓ | | ARM64e Apple iOS
[`arm64e-apple-tvos`](platform-support/arm64e-apple-tvos.md)  | ✓ | | ARM64e Apple tvOS
[`armeb-unknown-linux-gnueabi`](platform-support/armeb-unknown-linux-gnueabi.md) | ✓ | ? | Arm BE8 the default Arm big-endian architecture since [Armv6](https://developer.arm.com/documentation/101754/0616/armlink-Reference/armlink-Command-line-Options/--be8?lang=en).
[`armv4t-none-eabi`](platform-support/armv4t-none-eabi.md) | * |  | Bare Armv4T
`armv4t-unknown-linux-gnueabi` | ? |  | Armv4T Linux
[`armv5te-none-eabi`](platform-support/armv5te-none-eabi.md) | * |  | Bare Armv5TE
`armv5te-unknown-linux-uclibceabi` | ? |  | Armv5TE Linux with uClibc
[`armv6-unknown-freebsd`](platform-support/freebsd.md) | ✓ | ✓ | Armv6 FreeBSD
[`armv6-unknown-netbsd-eabihf`](platform-support/netbsd.md) | ✓ | ✓ | Armv6 NetBSD w/hard-float
[`armv6k-nintendo-3ds`](platform-support/armv6k-nintendo-3ds.md) | ? |  | Armv6k Nintendo 3DS, Horizon (Requires devkitARM toolchain)
[`armv7-rtems-eabihf`](platform-support/armv7-rtems-eabihf.md) | ? |  | RTEMS OS for ARM BSPs
[`armv7-sony-vita-newlibeabihf`](platform-support/armv7-sony-vita-newlibeabihf.md) | ✓ |  | Armv7-A Cortex-A9 Sony PlayStation Vita (requires VITASDK toolchain)
[`armv7-unknown-freebsd`](platform-support/freebsd.md) | ✓ | ✓ | Armv7-A FreeBSD
[`armv7-unknown-linux-uclibceabi`](platform-support/armv7-unknown-linux-uclibceabi.md) | ✓ | ✓ | Armv7-A Linux with uClibc, softfloat
[`armv7-unknown-linux-uclibceabihf`](platform-support/armv7-unknown-linux-uclibceabihf.md) | ✓ | ? | Armv7-A Linux with uClibc, hardfloat
[`armv7-unknown-netbsd-eabihf`](platform-support/netbsd.md) | ✓ | ✓ | Armv7-A NetBSD w/hard-float
[`armv7-unknown-trusty`](platform-support/trusty.md) | ✓ |  |
[`armv7-wrs-vxworks-eabihf`](platform-support/vxworks.md) | ✓ |  | Armv7-A for VxWorks
[`armv7a-kmc-solid_asp3-eabi`](platform-support/kmc-solid.md) | ✓ |  | ARM SOLID with TOPPERS/ASP3
[`armv7a-kmc-solid_asp3-eabihf`](platform-support/kmc-solid.md) | ✓ |  | ARM SOLID with TOPPERS/ASP3, hardfloat
[`armv7a-none-eabihf`](platform-support/arm-none-eabi.md) | * |  | Bare Armv7-A, hardfloat
[`armv7k-apple-watchos`](platform-support/apple-watchos.md) | ✓ |  | Armv7-A Apple WatchOS
[`armv7s-apple-ios`](platform-support/apple-ios.md) | ✓ |  | Armv7-A Apple-A6 Apple iOS
[`armv8r-none-eabihf`](platform-support/armv8r-none-eabihf.md) | * |  | Bare Armv8-R, hardfloat
[`armv7a-nuttx-eabi`](platform-support/nuttx.md) | ✓ |  | ARMv7-A with NuttX
[`armv7a-nuttx-eabihf`](platform-support/nuttx.md) | ✓ |  | ARMv7-A with NuttX, hardfloat
[`avr-none`](platform-support/avr-none.md) | * |  | AVR; requires `-Zbuild-std=core` and `-Ctarget-cpu=...`
`bpfeb-unknown-none` | * |  | BPF (big endian)
`bpfel-unknown-none` | * |  | BPF (little endian)
`csky-unknown-linux-gnuabiv2` | ✓ |  | C-SKY abiv2 Linux (little endian)
`csky-unknown-linux-gnuabiv2hf` | ✓ |  | C-SKY abiv2 Linux, hardfloat (little endian)
[`hexagon-unknown-linux-musl`](platform-support/hexagon-unknown-linux-musl.md) | ✓ | | Hexagon Linux with musl 1.2.3
[`hexagon-unknown-none-elf`](platform-support/hexagon-unknown-none-elf.md)| * | | Bare Hexagon (v60+, HVX)
[`i386-apple-ios`](platform-support/apple-ios.md) | ✓ |  | 32-bit x86 iOS (Penryn) [^x86_32-floats-return-ABI]
[`i586-unknown-netbsd`](platform-support/netbsd.md) | ✓ |  | 32-bit x86 (original Pentium) [^x86_32-floats-x87]
[`i586-unknown-redox`](platform-support/redox.md) | ✓ |  | 32-bit x86 Redox OS (PentiumPro) [^x86_32-floats-x87]
[`i686-apple-darwin`](platform-support/apple-darwin.md) | ✓ | ✓ | 32-bit macOS (10.12+, Sierra+, Penryn) [^x86_32-floats-return-ABI]
[`i686-pc-nto-qnx700`](platform-support/nto-qnx.md) | * |  | 32-bit x86 QNX Neutrino 7.0 RTOS (Pentium 4) [^x86_32-floats-return-ABI]
`i686-unknown-haiku` | ✓ | ✓ | 32-bit Haiku (Pentium 4) [^x86_32-floats-return-ABI]
[`i686-unknown-hurd-gnu`](platform-support/hurd.md) | ✓ | ✓ | 32-bit GNU/Hurd (Pentium 4) [^x86_32-floats-return-ABI]
[`i686-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | NetBSD/i386 (Pentium 4) [^x86_32-floats-return-ABI]
[`i686-unknown-openbsd`](platform-support/openbsd.md) | ✓ | ✓ | 32-bit OpenBSD (Pentium 4) [^x86_32-floats-return-ABI]
`i686-uwp-windows-gnu` | ✓ |  | [^x86_32-floats-return-ABI]
[`i686-uwp-windows-msvc`](platform-support/uwp-windows-msvc.md) | ✓ |  | [^x86_32-floats-return-ABI] [^win32-msvc-alignment]
[`i686-win7-windows-gnu`](platform-support/win7-windows-gnu.md) | ✓ |   | 32-bit Windows 7 support [^x86_32-floats-return-ABI]
[`i686-win7-windows-msvc`](platform-support/win7-windows-msvc.md) | ✓ |   | 32-bit Windows 7 support [^x86_32-floats-return-ABI] [^win32-msvc-alignment]
[`i686-wrs-vxworks`](platform-support/vxworks.md) | ✓ |  | [^x86_32-floats-return-ABI]
[`loongarch64-unknown-linux-ohos`](platform-support/openharmony.md) | ✓ |   | LoongArch64 OpenHarmony
[`loongarch32-unknown-none`](platform-support/loongarch-none.md) | * | LoongArch32 Bare-metal (ILP32D ABI)
[`loongarch32-unknown-none-softfloat`](platform-support/loongarch-none.md) | * | LoongArch32 Bare-metal (ILP32S ABI)
[`m68k-unknown-linux-gnu`](platform-support/m68k-unknown-linux-gnu.md) | ? |  | Motorola 680x0 Linux
[`m68k-unknown-none-elf`](platform-support/m68k-unknown-none-elf.md) |  |  | Motorola 680x0
`mips-unknown-linux-gnu` | ✓ | ✓ | MIPS Linux (kernel 4.4, glibc 2.23)
`mips-unknown-linux-musl` | ✓ |  | MIPS Linux with musl 1.2.3
`mips-unknown-linux-uclibc` | ✓ |  | MIPS Linux with uClibc
[`mips64-openwrt-linux-musl`](platform-support/mips64-openwrt-linux-musl.md) | ? |  | MIPS64 for OpenWrt Linux musl 1.2.3
`mips64-unknown-linux-gnuabi64` | ✓ | ✓ | MIPS64 Linux, N64 ABI (kernel 4.4, glibc 2.23)
`mips64-unknown-linux-muslabi64` | ✓ |  | MIPS64 Linux, N64 ABI, musl 1.2.3
`mips64el-unknown-linux-gnuabi64` | ✓ | ✓ | MIPS64 (little endian) Linux, N64 ABI (kernel 4.4, glibc 2.23)
`mips64el-unknown-linux-muslabi64` | ✓ |  | MIPS64 (little endian) Linux, N64 ABI, musl 1.2.3
`mipsel-sony-psp` | * |  | MIPS (LE) Sony PlayStation Portable (PSP)
[`mipsel-sony-psx`](platform-support/mipsel-sony-psx.md) | * |  | MIPS (LE) Sony PlayStation 1 (PSX)
[`mipsel-unknown-linux-gnu`](platform-support/mipsel-unknown-linux-gnu.md) | ✓ | ✓ | MIPS (little endian) Linux (kernel 4.4, glibc 2.23)
`mipsel-unknown-linux-musl` | ✓ |  | MIPS (little endian) Linux with musl 1.2.3
`mipsel-unknown-linux-uclibc` | ✓ |  | MIPS (LE) Linux with uClibc
[`mipsel-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | 32-bit MIPS (LE), requires mips32 cpu support
`mipsel-unknown-none` | * |  | Bare MIPS (LE) softfloat
[`mips-mti-none-elf`](platform-support/mips-mti-none-elf.md) | * |  | Bare MIPS32r2 (BE) softfloat
[`mipsel-mti-none-elf`](platform-support/mips-mti-none-elf.md) | * |  | Bare MIPS32r2 (LE) softfloat
[`mipsisa32r6-unknown-linux-gnu`](platform-support/mips-release-6.md) | ? |  | 32-bit MIPS Release 6 Big Endian
[`mipsisa32r6el-unknown-linux-gnu`](platform-support/mips-release-6.md) | ? |  | 32-bit MIPS Release 6 Little Endian
[`mipsisa64r6-unknown-linux-gnuabi64`](platform-support/mips-release-6.md) | ? |  | 64-bit MIPS Release 6 Big Endian
[`mipsisa64r6el-unknown-linux-gnuabi64`](platform-support/mips-release-6.md) | ✓ | ✓ | 64-bit MIPS Release 6 Little Endian
`msp430-none-elf` | * |  | 16-bit MSP430 microcontrollers
[`powerpc-unknown-freebsd`](platform-support/freebsd.md) | ? |   | PowerPC FreeBSD
[`powerpc-unknown-linux-gnuspe`](platform-support/powerpc-unknown-linux-gnuspe.md) | ✓ |  | PowerPC SPE Linux
`powerpc-unknown-linux-musl` | ? |  | PowerPC Linux with musl 1.2.3
[`powerpc-unknown-linux-muslspe`](platform-support/powerpc-unknown-linux-muslspe.md) | ? |  | PowerPC SPE Linux with musl 1.2.3
[`powerpc-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | NetBSD 32-bit powerpc systems
[`powerpc-unknown-openbsd`](platform-support/powerpc-unknown-openbsd.md) | * |  |
[`powerpc-wrs-vxworks`](platform-support/vxworks.md) | ✓ |  |
[`powerpc-wrs-vxworks-spe`](platform-support/vxworks.md) | ✓ |  |
[`powerpc64-ibm-aix`](platform-support/aix.md) | ? |  | 64-bit AIX (7.2 and newer)
[`powerpc64-unknown-freebsd`](platform-support/freebsd.md) | ✓ | ✓ | PPC64 FreeBSD (ELFv2)
[`powerpc64-unknown-linux-musl`](platform-support/powerpc64-unknown-linux-musl.md) | ✓ | ✓ | PPC64 Linux (kernel 4.19, musl 1.2.3)
[`powerpc64-unknown-openbsd`](platform-support/openbsd.md) | ✓ | ✓ | OpenBSD/powerpc64
[`powerpc64-wrs-vxworks`](platform-support/vxworks.md) | ✓ |  |
[`powerpc64le-unknown-freebsd`](platform-support/freebsd.md) | ✓ | ✓ | PPC64LE FreeBSD
[`riscv32-wrs-vxworks`](platform-support/vxworks.md) | ✓ |  |
[`riscv32e-unknown-none-elf`](platform-support/riscv32e-unknown-none-elf.md) | * |  | Bare RISC-V (RV32E ISA)
[`riscv32em-unknown-none-elf`](platform-support/riscv32e-unknown-none-elf.md) | * |  | Bare RISC-V (RV32EM ISA)
[`riscv32emc-unknown-none-elf`](platform-support/riscv32e-unknown-none-elf.md) | * |  | Bare RISC-V (RV32EMC ISA)
`riscv32gc-unknown-linux-gnu` | ✓ |   | RISC-V Linux (kernel 5.4, glibc 2.33)
`riscv32gc-unknown-linux-musl` | ? |   | RISC-V Linux (kernel 5.4, musl 1.2.3 + RISCV32 support patches)
[`riscv32im-risc0-zkvm-elf`](platform-support/riscv32im-risc0-zkvm-elf.md) | ? |  | RISC Zero's zero-knowledge Virtual Machine (RV32IM ISA)
[`riscv32ima-unknown-none-elf`](platform-support/riscv32-unknown-none-elf.md) | * |  | Bare RISC-V (RV32IMA ISA)
[`riscv32imac-esp-espidf`](platform-support/esp-idf.md) | ✓ |  | RISC-V ESP-IDF
[`riscv32imac-unknown-nuttx-elf`](platform-support/nuttx.md) | ✓ |  | RISC-V 32bit with NuttX
[`riscv32imac-unknown-xous-elf`](platform-support/riscv32imac-unknown-xous-elf.md) | ? |  | RISC-V Xous (RV32IMAC ISA)
[`riscv32imafc-esp-espidf`](platform-support/esp-idf.md) | ✓ |  | RISC-V ESP-IDF
[`riscv32imafc-unknown-nuttx-elf`](platform-support/nuttx.md) | ✓ |  | RISC-V 32bit with NuttX
[`riscv32imc-esp-espidf`](platform-support/esp-idf.md) | ✓ |  | RISC-V ESP-IDF
[`riscv32imc-unknown-nuttx-elf`](platform-support/nuttx.md) | ✓ |  | RISC-V 32bit with NuttX
[`riscv64-linux-android`](platform-support/android.md) | ? |   | RISC-V 64-bit Android
[`riscv64-wrs-vxworks`](platform-support/vxworks.md) | ✓ |  |
`riscv64gc-unknown-freebsd` | ? |   | RISC-V FreeBSD
`riscv64gc-unknown-fuchsia` | ? |   | RISC-V Fuchsia
[`riscv64gc-unknown-hermit`](platform-support/hermit.md) | ✓ |   | RISC-V Hermit
[`riscv64gc-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | RISC-V NetBSD
[`riscv64gc-unknown-nuttx-elf`](platform-support/nuttx.md) | ✓ |  | RISC-V 64bit with NuttX
[`riscv64gc-unknown-openbsd`](platform-support/openbsd.md) | ✓ | ✓ | OpenBSD/riscv64
[`riscv64imac-unknown-nuttx-elf`](platform-support/nuttx.md) | ✓ |  | RISC-V 64bit with NuttX
[`s390x-unknown-linux-musl`](platform-support/s390x-unknown-linux-musl.md) | ✓ |  | S390x Linux (kernel 3.2, musl 1.2.3)
`sparc-unknown-linux-gnu` | ✓ |  | 32-bit SPARC Linux
[`sparc-unknown-none-elf`](./platform-support/sparc-unknown-none-elf.md) | * |  | Bare 32-bit SPARC V7+
[`sparc64-unknown-netbsd`](platform-support/netbsd.md) | ✓ | ✓ | NetBSD/sparc64
[`sparc64-unknown-openbsd`](platform-support/openbsd.md) | ✓ | ✓ | OpenBSD/sparc64
[`thumbv4t-none-eabi`](platform-support/armv4t-none-eabi.md) | * |  | Thumb-mode Bare Armv4T
[`thumbv5te-none-eabi`](platform-support/armv5te-none-eabi.md) | * |  | Thumb-mode Bare Armv5TE
[`thumbv6m-nuttx-eabi`](platform-support/nuttx.md) | ✓ |  | ARMv6M with NuttX
`thumbv7a-pc-windows-msvc` |  |  |
[`thumbv7a-uwp-windows-msvc`](platform-support/uwp-windows-msvc.md) |  |  |
[`thumbv7a-nuttx-eabi`](platform-support/nuttx.md) | ✓ |  | ARMv7-A with NuttX
[`thumbv7a-nuttx-eabihf`](platform-support/nuttx.md) | ✓ |  | ARMv7-A with NuttX, hardfloat
[`thumbv7em-nuttx-eabi`](platform-support/nuttx.md) | ✓ |  | ARMv7EM with NuttX
[`thumbv7em-nuttx-eabihf`](platform-support/nuttx.md) | ✓ |  | ARMv7EM with NuttX, hardfloat
[`thumbv7m-nuttx-eabi`](platform-support/nuttx.md) | ✓ |  | ARMv7M with NuttX
`thumbv7neon-unknown-linux-musleabihf` | ? |  | Thumb2-mode Armv7-A Linux with NEON, musl 1.2.3
[`thumbv8m.base-nuttx-eabi`](platform-support/nuttx.md) | ✓ |  | ARMv8M Baseline with NuttX
[`thumbv8m.main-nuttx-eabi`](platform-support/nuttx.md) | ✓ |  | ARMv8M Mainline with NuttX
[`thumbv8m.main-nuttx-eabihf`](platform-support/nuttx.md) | ✓ |  | ARMv8M Mainline with NuttX, hardfloat
[`wasm64-unknown-unknown`](platform-support/wasm64-unknown-unknown.md) | ? |  | WebAssembly
[`wasm32-wali-linux-musl`](platform-support/wasm32-wali-linux.md) | ? |  | WebAssembly with [WALI](https://github.com/arjunr2/WALI)
[`x86_64-apple-tvos`](platform-support/apple-tvos.md) | ✓ |  | x86 64-bit tvOS
[`x86_64-apple-watchos-sim`](platform-support/apple-watchos.md) | ✓ |  | x86 64-bit Apple WatchOS simulator
[`x86_64-lynx-lynxos178`](platform-support/lynxos178.md) |   |  | x86_64 LynxOS-178
[`x86_64-pc-cygwin`](platform-support/x86_64-pc-cygwin.md) | ✓ |  | 64-bit x86 Cygwin |
[`x86_64-pc-nto-qnx710`](platform-support/nto-qnx.md) | ✓ |  | x86 64-bit QNX Neutrino 7.1 RTOS with default network stack (io-pkt) |
[`x86_64-pc-nto-qnx710_iosock`](platform-support/nto-qnx.md) | ✓ |  | x86 64-bit QNX Neutrino 7.1 RTOS with new network stack (io-sock) |
[`x86_64-pc-nto-qnx800`](platform-support/nto-qnx.md) | ✓ |  | x86 64-bit QNX Neutrino 8.0 RTOS |
[`x86_64-unikraft-linux-musl`](platform-support/unikraft-linux-musl.md) | ✓ |   | 64-bit Unikraft with musl 1.2.3
`x86_64-unknown-dragonfly` | ✓ | ✓ | 64-bit DragonFlyBSD
`x86_64-unknown-haiku` | ✓ | ✓ | 64-bit Haiku
[`x86_64-unknown-hermit`](platform-support/hermit.md) | ✓ |  | x86_64 Hermit
[`x86_64-unknown-hurd-gnu`](platform-support/hurd.md) | ✓ | ✓ | 64-bit GNU/Hurd
`x86_64-unknown-l4re-uclibc` | ? |  |
[`x86_64-unknown-linux-none`](platform-support/x86_64-unknown-linux-none.md) | * |  | 64-bit Linux with no libc
[`x86_64-unknown-openbsd`](platform-support/openbsd.md) | ✓ | ✓ | 64-bit OpenBSD
[`x86_64-unknown-trusty`](platform-support/trusty.md) | ✓ |  |
`x86_64-uwp-windows-gnu` | ✓ |  |
[`x86_64-uwp-windows-msvc`](platform-support/uwp-windows-msvc.md) | ✓ |  |
[`x86_64-win7-windows-gnu`](platform-support/win7-windows-gnu.md) | ✓ |   | 64-bit Windows 7 support
[`x86_64-win7-windows-msvc`](platform-support/win7-windows-msvc.md) | ✓ |   | 64-bit Windows 7 support
[`x86_64-wrs-vxworks`](platform-support/vxworks.md) | ✓ |  |
[`x86_64h-apple-darwin`](platform-support/x86_64h-apple-darwin.md) | ✓ | ✓ | macOS with late-gen Intel (at least Haswell)
[`xtensa-esp32-espidf`](platform-support/esp-idf.md) | ✓ |  | Xtensa ESP32
[`xtensa-esp32-none-elf`](platform-support/xtensa.md) | * |  | Xtensa ESP32
[`xtensa-esp32s2-espidf`](platform-support/esp-idf.md) | ✓ |  | Xtensa ESP32-S2
[`xtensa-esp32s2-none-elf`](platform-support/xtensa.md) | * |  | Xtensa ESP32-S2
[`xtensa-esp32s3-espidf`](platform-support/esp-idf.md) | ✓ |  | Xtensa ESP32-S3
[`xtensa-esp32s3-none-elf`](platform-support/xtensa.md) | * |  | Xtensa ESP32-S3

[runs on NVIDIA GPUs]: https://github.com/japaric-archived/nvptx#targets
[the AMD GPU]: https://llvm.org/docs/AMDGPUUsage.html#processors
