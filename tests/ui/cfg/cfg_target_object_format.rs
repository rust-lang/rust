//@ add-minicore
//@ check-pass
//@ ignore-backends: gcc
//
//@ revisions: linux_gnu linux_musl linux_ohos linux_powerpc
//@[linux_gnu] compile-flags: --target aarch64-unknown-linux-gnu
//@[linux_gnu] needs-llvm-components: aarch64
//@[linux_musl] compile-flags: --target aarch64-unknown-linux-musl
//@[linux_musl] needs-llvm-components: aarch64
//@[linux_ohos] compile-flags: --target aarch64-unknown-linux-ohos
//@[linux_ohos] needs-llvm-components: aarch64
//@[linux_powerpc] compile-flags: --target powerpc-unknown-linux-gnu
//@[linux_powerpc] needs-llvm-components: powerpc
//
//@ revisions: darwin ios
//@[darwin] compile-flags: --target aarch64-apple-darwin
//@[darwin] needs-llvm-components: aarch64
//@[ios] compile-flags: --target aarch64-apple-ios
//@[ios] needs-llvm-components: aarch64
//
//@ revisions: win_msvc win_gnu
//@[win_msvc] compile-flags: --target aarch64-pc-windows-msvc
//@[win_msvc] needs-llvm-components: aarch64
//@[win_gnu] compile-flags: --target x86_64-pc-windows-gnu
//@[win_gnu] needs-llvm-components: x86
//
//@ revisions: wasm32 wasm64
//@[wasm32] compile-flags: --target wasm32-unknown-unknown
//@[wasm32] needs-llvm-components: webassembly
//@[wasm64] compile-flags: --target wasm64-unknown-unknown
//@[wasm64] needs-llvm-components: webassembly
//
//@ revisions: aix
//@[aix] compile-flags: --target powerpc64-ibm-aix
//@[aix] needs-llvm-components: powerpc
//
//@ revisions: hermit sgx uefi
//@[hermit] compile-flags: --target x86_64-unknown-hermit
//@[hermit] needs-llvm-components: x86
//@[sgx] compile-flags: --target x86_64-fortanix-unknown-sgx
//@[sgx] needs-llvm-components: x86
//@[uefi] compile-flags: --target x86_64-unknown-uefi
//@[uefi] needs-llvm-components: x86
//
//@ revisions: bpfeb bpfel
//@[bpfeb] compile-flags: --target bpfeb-unknown-none
//@[bpfeb] needs-llvm-components: bpf
//@[bpfel] compile-flags: --target bpfel-unknown-none
//@[bpfel] needs-llvm-components: bpf
//
//@ revisions: avr
//@[avr] compile-flags: --target avr-none -Ctarget-cpu=atmega328
//@[avr] needs-llvm-components: avr
//
//@ revisions: msp430
//@[msp430] compile-flags: --target msp430-none-elf
//@[msp430] needs-llvm-components: msp430
//
//@ revisions: thumb
//@[thumb] compile-flags: --target thumbv7m-none-eabi
//@[thumb] needs-llvm-components: arm
#![crate_type = "lib"]
#![feature(no_core, lang_items, cfg_target_object_format)]
#![no_core]

extern crate minicore;
use minicore::*;

macro_rules! assert_cfg {
    ($rhs:ident = $rhs_val:literal) => {
        #[cfg(not($rhs = $rhs_val))]
        compile_error!(concat!("expected `", stringify!($rhs), " = ", $rhs_val, "`",));
    };
}

const _: () = {
    cfg_select!(
        target_os = "linux" => assert_cfg!(target_object_format = "elf"),
        target_os = "aix" => assert_cfg!(target_object_format = "xcoff"),
        target_os = "uefi" => assert_cfg!(target_object_format = "coff"),
        target_os = "windows" => assert_cfg!(target_object_format = "coff"),
        target_os = "hermit" => assert_cfg!(target_object_format = "elf"),

        target_arch = "bpf" => assert_cfg!(target_object_format = "elf"),
        target_arch = "avr" => assert_cfg!(target_object_format = "elf"),
        target_arch = "msp430" => assert_cfg!(target_object_format = "elf"),

        target_abi = "eabi" => assert_cfg!(target_object_format = "elf"),
        target_vendor = "apple" => assert_cfg!(target_object_format = "mach-o"),
        target_family = "wasm" => assert_cfg!(target_object_format = "wasm"),

        windows => assert_cfg!(target_object_format = "coff"),

        _ => {}
    );
};

const _: () = {
    cfg_select!(
        target_object_format = "mach-o" => assert_cfg!(target_vendor = "apple"),
        target_object_format = "wasm" => assert_cfg!(target_family = "wasm"),
        target_object_format = "xcoff" => assert_cfg!(target_os = "aix"),
        _ => {}
    );
};
