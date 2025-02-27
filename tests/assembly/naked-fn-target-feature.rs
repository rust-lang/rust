//@ revisions: aarch64-elf aarch64-macho aarch64-coff x86_64 s390x riscv64 powerpc64 loongarch64
//@ add-core-stubs
//@ assembly-output: emit-asm
//
//@ [x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@ [x86_64] needs-llvm-components: x86
//
//@ [aarch64-elf] compile-flags: --target aarch64-unknown-linux-gnu
//@ [aarch64-elf] needs-llvm-components: aarch64
//@ [aarch64-macho] compile-flags: --target aarch64-apple-darwin
//@ [aarch64-macho] needs-llvm-components: aarch64
//@ [aarch64-coff] compile-flags: --target aarch64-pc-windows-gnullvm
//@ [aarch64-coff] needs-llvm-components: aarch64
//
//@ [s390x] compile-flags: --target s390x-unknown-linux-gnu
//@ [s390x] needs-llvm-components: systemz
//
//@ [powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//@ [powerpc64] needs-llvm-components: powerpc
//
//@ [riscv64] compile-flags: --target riscv64gc-unknown-linux-gnu
//@ [riscv64] needs-llvm-components: riscv
//
// NOTE: there currently no logic for handling target features for loongarch,
// because it does not seem to support a feature-setting directive.
// we effectively assume all instructions are accepted regardless of target feature.
//@ [loongarch64] compile-flags: --target loongarch64-unknown-linux-gnu
//@ [loongarch64] needs-llvm-components: loongarch
//
// NOTE: wasm32 is skipped because it does not work
// [wasm32] compile-flags: --target wasm32-wasip1
// [wasm32] needs-llvm-components: webassembly

// Test that the #[target_feature(enable = ...)]` works on naked functions.
//
// For most targets, a directive needs to be applied to enable, and then disable the target feature.

#![crate_type = "lib"]
#![feature(no_core, naked_functions, asm_experimental_arch)]
#![feature(
    avx512_target_feature,
    s390x_target_feature,
    powerpc_target_feature,
    loongarch_target_feature,
    m68k_target_feature
)]
#![no_core]

extern crate minicore;
use minicore::*;

// x86_64-LABEL: vpclmulqdq:
// x86_64: vpclmulqdq
#[no_mangle]
#[naked]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "vpclmulqdq")]
unsafe extern "C" fn vpclmulqdq() {
    naked_asm!("vpclmulqdq zmm1, zmm2, zmm3, 4")
}

// i8mm is not enabled by default
//
// note that aarch64-apple-darwin enables more features than aarch64-unknown-linux-gnu
//
// aarch64-elf-LABEL: i8mm:
// aarch64-elf: usdot
// aarch64-macho-LABEL: i8mm:
// aarch64-macho: usdot
// aarch64-coff-LABEL: i8mm:
// aarch64-coff: usdot
#[no_mangle]
#[naked]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
unsafe extern "C" fn i8mm() {
    naked_asm!("usdot   v0.4s, v1.16b, v2.4b[3]")
}

// riscv64: sh1add:
// riscv64: sh1add
#[no_mangle]
#[naked]
#[cfg(target_arch = "riscv64")]
#[target_feature(enable = "zba")]
unsafe extern "C" fn sh1add() {
    naked_asm!("sh1add a0, a1, a2", "ret");
}

#[cfg(target_arch = "s390x")]
mod s390x {
    use super::*;

    // s390x: vector:
    // s390x: vavglg
    #[no_mangle]
    #[naked]
    #[target_feature(enable = "vector")]
    unsafe extern "C" fn vector() {
        naked_asm!("vavglg  %v0, %v0, %v0")
    }

    // s390x: vector_enhancements_1:
    // s390x: vfcesbs
    #[no_mangle]
    #[naked]
    #[target_feature(enable = "vector-enhancements-1")]
    unsafe extern "C" fn vector_enhancements_1() {
        naked_asm!("vfcesbs %v0, %v0, %v0")
    }

    // s390x: vector_enhancements_2:
    // s390x: vclfp
    #[no_mangle]
    #[naked]
    #[target_feature(enable = "vector-enhancements-2")]
    unsafe extern "C" fn vector_enhancements_2() {
        naked_asm!("vclfp   %v0, %v0, 0, 0, 0")
    }

    // s390x: vector_packed_decimal:
    // s390x: vlrlr
    #[no_mangle]
    #[naked]
    #[target_feature(enable = "vector-packed-decimal")]
    unsafe extern "C" fn vector_packed_decimal() {
        naked_asm!("vlrlr   %v24, %r3, 0(%r2)", "br      %r14")
    }

    // s390x: vector_packed_decimal_enhancement:
    // s390x: vcvbg
    #[no_mangle]
    #[naked]
    #[target_feature(enable = "vector-packed-decimal-enhancement")]
    unsafe extern "C" fn vector_packed_decimal_enhancement() {
        naked_asm!("vcvbg   %r0, %v0, 0, 1")
    }

    // s390x: vector_packed_decimal_enhancement_2:
    // s390x: vupkzl
    #[no_mangle]
    #[naked]
    #[target_feature(enable = "vector-packed-decimal-enhancement-2")]
    unsafe extern "C" fn vector_packed_decimal_enhancement_2() {
        naked_asm!("vupkzl  %v0, %v0, 0")
    }
}

// powerpc64: power10_vector:
// powerpc64: xxpermx
#[no_mangle]
#[naked]
#[cfg(target_arch = "powerpc64")]
#[target_feature(enable = "power10-vector")]
unsafe extern "C" fn power10_vector() {
    naked_asm!("xxpermx 34, 0, 1, 2, 0", "blr")
}

// loongarch64: lasx:
// loongarch64: xvadd.b
#[no_mangle]
#[naked]
#[cfg(target_arch = "loongarch64")]
#[target_feature(enable = "lasx")]
unsafe extern "C" fn lasx() {
    naked_asm!("xvadd.b  $xr0, $xr0, $xr1", "ret")
}

// wasm32: simd128:
// wasm32: i8x16.shuffle
#[no_mangle]
#[naked]
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe extern "C" fn simd128() {
    naked_asm!("i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15", "return");
}
