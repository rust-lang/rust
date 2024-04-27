//@ assembly-output: emit-asm
// ignore-tidy-linelength
//@ revisions: aarch64_pc_windows_msvc
//@ [aarch64_pc_windows_msvc] compile-flags: --target aarch64-pc-windows-msvc
//@ [aarch64_pc_windows_msvc] needs-llvm-components: aarch64
//@ revisions: aarch64_pc_windows_gnullvm
//@ [aarch64_pc_windows_gnullvm] compile-flags: --target aarch64-pc-windows-gnullvm
//@ [aarch64_pc_windows_gnullvm] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_uefi
//@ [aarch64_unknown_uefi] compile-flags: --target aarch64-unknown-uefi
//@ [aarch64_unknown_uefi] needs-llvm-components: aarch64
//@ revisions: aarch64_uwp_windows_msvc
//@ [aarch64_uwp_windows_msvc] compile-flags: --target aarch64-uwp-windows-msvc
//@ [aarch64_uwp_windows_msvc] needs-llvm-components: aarch64
//@ revisions: arm64ec_pc_windows_msvc
//@ [arm64ec_pc_windows_msvc] compile-flags: --target arm64ec-pc-windows-msvc
//@ [arm64ec_pc_windows_msvc] needs-llvm-components: aarch64
//@ revisions: avr_unknown_gnu_atmega328
//@ [avr_unknown_gnu_atmega328] compile-flags: --target avr-unknown-gnu-atmega328
//@ [avr_unknown_gnu_atmega328] needs-llvm-components: avr
//@ revisions: bpfeb_unknown_none
//@ [bpfeb_unknown_none] compile-flags: --target bpfeb-unknown-none
//@ [bpfeb_unknown_none] needs-llvm-components: bpf
//@ revisions: bpfel_unknown_none
//@ [bpfel_unknown_none] compile-flags: --target bpfel-unknown-none
//@ [bpfel_unknown_none] needs-llvm-components: bpf
//@ revisions: i586_pc_windows_msvc
//@ [i586_pc_windows_msvc] compile-flags: --target i586-pc-windows-msvc
//@ [i586_pc_windows_msvc] needs-llvm-components: x86
//@ revisions: i686_pc_windows_gnu
//@ [i686_pc_windows_gnu] compile-flags: --target i686-pc-windows-gnu
//@ [i686_pc_windows_gnu] needs-llvm-components: x86
//@ revisions: i686_pc_windows_msvc
//@ [i686_pc_windows_msvc] compile-flags: --target i686-pc-windows-msvc
//@ [i686_pc_windows_msvc] needs-llvm-components: x86
//@ revisions: i686_pc_windows_gnullvm
//@ [i686_pc_windows_gnullvm] compile-flags: --target i686-pc-windows-gnullvm
//@ [i686_pc_windows_gnullvm] needs-llvm-components: x86
//@ revisions: i686_uwp_windows_gnu
//@ [i686_uwp_windows_gnu] compile-flags: --target i686-uwp-windows-gnu
//@ [i686_uwp_windows_gnu] needs-llvm-components: x86
//@ revisions: i686_unknown_uefi
//@ [i686_unknown_uefi] compile-flags: --target i686-unknown-uefi
//@ [i686_unknown_uefi] needs-llvm-components: x86
//@ revisions: i686_uwp_windows_msvc
//@ [i686_uwp_windows_msvc] compile-flags: --target i686-uwp-windows-msvc
//@ [i686_uwp_windows_msvc] needs-llvm-components: x86
//@ revisions: i686_win7_windows_msvc
//@ [i686_win7_windows_msvc] compile-flags: --target i686-win7-windows-msvc
//@ [i686_win7_windows_msvc] needs-llvm-components: x86
//@ revisions: powerpc64_ibm_aix
//@ [powerpc64_ibm_aix] compile-flags: --target powerpc64-ibm-aix
//@ [powerpc64_ibm_aix] needs-llvm-components: powerpc
//@ revisions: thumbv7a_uwp_windows_msvc
//@ [thumbv7a_uwp_windows_msvc] compile-flags: --target thumbv7a-uwp-windows-msvc
//@ [thumbv7a_uwp_windows_msvc] needs-llvm-components: arm
//@ revisions: thumbv7a_pc_windows_msvc
//@ [thumbv7a_pc_windows_msvc] compile-flags: --target thumbv7a-pc-windows-msvc
//@ [thumbv7a_pc_windows_msvc] needs-llvm-components: arm
//@ revisions: x86_64_pc_windows_gnu
//@ [x86_64_pc_windows_gnu] compile-flags: --target x86_64-pc-windows-gnu
//@ [x86_64_pc_windows_gnu] needs-llvm-components: x86
//@ revisions: x86_64_pc_windows_gnullvm
//@ [x86_64_pc_windows_gnullvm] compile-flags: --target x86_64-pc-windows-gnullvm
//@ [x86_64_pc_windows_gnullvm] needs-llvm-components: x86
//@ revisions: x86_64_pc_windows_msvc
//@ [x86_64_pc_windows_msvc] compile-flags: --target x86_64-pc-windows-msvc
//@ [x86_64_pc_windows_msvc] needs-llvm-components: x86
//@ revisions: x86_64_unknown_uefi
//@ [x86_64_unknown_uefi] compile-flags: --target x86_64-unknown-uefi
//@ [x86_64_unknown_uefi] needs-llvm-components: x86
//@ revisions: x86_64_uwp_windows_gnu
//@ [x86_64_uwp_windows_gnu] compile-flags: --target x86_64-uwp-windows-gnu
//@ [x86_64_uwp_windows_gnu] needs-llvm-components: x86
//@ revisions: x86_64_uwp_windows_msvc
//@ [x86_64_uwp_windows_msvc] compile-flags: --target x86_64-uwp-windows-msvc
//@ [x86_64_uwp_windows_msvc] needs-llvm-components: x86
//@ revisions: x86_64_win7_windows_msvc
//@ [x86_64_win7_windows_msvc] compile-flags: --target x86_64-win7-windows-msvc
//@ [x86_64_win7_windows_msvc] needs-llvm-components: x86

// Sanity-check that each target can produce assembly code.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

#[lang = "sized"]
trait Sized {}

pub fn test() -> u8 {
    42
}

// CHECK: .file
