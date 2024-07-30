// verify that simd mask reductions do not introduce additional bit shift operations
//@ revisions: x86 aarch64
//@ [x86] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
// Set the base cpu explicitly, in case the default has been changed.
//@ [x86] compile-flags: -C target-cpu=x86-64
//@ [x86] needs-llvm-components: x86
//@ [aarch64] compile-flags: --target=aarch64-unknown-linux-gnu
//@ [aarch64] needs-llvm-components: aarch64
//@ [aarch64] min-llvm-version: 18.0
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -O -C panic=abort

#![feature(no_core, lang_items, repr_simd, intrinsics)]
#![no_core]
#![allow(non_camel_case_types)]

// Because we don't have core yet.
#[lang = "sized"]
pub trait Sized {}

#[lang = "copy"]
trait Copy {}

#[repr(simd)]
pub struct mask8x16([i8; 16]);

extern "rust-intrinsic" {
    fn simd_reduce_all<T>(x: T) -> bool;
    fn simd_reduce_any<T>(x: T) -> bool;
}

// CHECK-LABEL: mask_reduce_all:
#[no_mangle]
pub unsafe extern "C" fn mask_reduce_all(m: mask8x16) -> bool {
    // CHECK-X86: psllw xmm0, 7
    // CHECK-X86-NEXT: pmovmskb eax, xmm0
    // CHECK-X86-NEXT: {{cmp ax, -1|xor eax, 65535}}
    // CHECK-X86-NEXT: sete al
    //
    // CHECK-AARCH64: shl v0.16b, v0.16b, #7
    // CHECK-AARCH64-NEXT: cmlt v0.16b, v0.16b, #0
    // CHECK-AARCH64-NEXT: uminv b0, v0.16b
    // CHECK-AARCH64-NEXT: fmov [[REG:[a-z0-9]+]], s0
    // CHECK-AARCH64-NEXT: and w0, [[REG]], #0x1
    simd_reduce_all(m)
}

// CHECK-LABEL: mask_reduce_any:
#[no_mangle]
pub unsafe extern "C" fn mask_reduce_any(m: mask8x16) -> bool {
    // CHECK-X86: psllw xmm0, 7
    // CHECK-X86-NEXT: pmovmskb
    // CHECK-X86-NEXT: test eax, eax
    // CHECK-X86-NEXT: setne al
    //
    // CHECK-AARCH64: shl v0.16b, v0.16b, #7
    // CHECK-AARCH64-NEXT: cmlt v0.16b, v0.16b, #0
    // CHECK-AARCH64-NEXT: umaxv b0, v0.16b
    // CHECK-AARCH64-NEXT: fmov [[REG:[a-z0-9]+]], s0
    // CHECK-AARCH64-NEXT: and w0, [[REG]], #0x1
    simd_reduce_any(m)
}
