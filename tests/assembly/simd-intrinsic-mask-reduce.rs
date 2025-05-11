// verify that simd mask reductions do not introduce additional bit shift operations
//@ add-core-stubs
//@ revisions: x86 aarch64
//@ [x86] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
// Set the base cpu explicitly, in case the default has been changed.
//@ [x86] compile-flags: -C target-cpu=x86-64
//@ [x86] needs-llvm-components: x86
//@ [aarch64] compile-flags: --target=aarch64-unknown-linux-gnu
//@ [aarch64] needs-llvm-components: aarch64
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -C panic=abort

#![feature(no_core, lang_items, repr_simd, intrinsics)]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct mask8x16([i8; 16]);

#[rustc_intrinsic]
unsafe fn simd_reduce_all<T>(x: T) -> bool;
#[rustc_intrinsic]
unsafe fn simd_reduce_any<T>(x: T) -> bool;

// CHECK-LABEL: mask_reduce_all:
#[no_mangle]
pub unsafe extern "C" fn mask_reduce_all(m: mask8x16) -> bool {
    // x86-NOT: psllw
    // x86: pmovmskb eax, xmm0
    // x86-NEXT: {{cmp ax, -1|cmp eax, 65535|xor eax, 65535}}
    // x86-NEXT: sete al
    //
    // aarch64-NOT: shl
    // aarch64: cmge v0.16b, v0.16b, #0
    // aarch64-DAG: mov [[REG1:[a-z0-9]+]], #1
    // aarch64-DAG: umaxv b0, v0.16b
    // aarch64-NEXT: fmov [[REG2:[a-z0-9]+]], s0
    // aarch64-NEXT: bic w0, [[REG1]], [[REG2]]
    simd_reduce_all(m)
}

// CHECK-LABEL: mask_reduce_any:
#[no_mangle]
pub unsafe extern "C" fn mask_reduce_any(m: mask8x16) -> bool {
    // x86-NOT: psllw
    // x86: pmovmskb
    // x86-NEXT: test eax, eax
    // x86-NEXT: setne al
    //
    // aarch64-NOT: shl
    // aarch64: cmlt v0.16b, v0.16b, #0
    // aarch64-NEXT: umaxv b0, v0.16b
    // aarch64-NEXT: fmov [[REG:[a-z0-9]+]], s0
    // aarch64-NEXT: and w0, [[REG]], #0x1
    simd_reduce_any(m)
}
