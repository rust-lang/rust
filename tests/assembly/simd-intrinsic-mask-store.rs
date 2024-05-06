//@ revisions: x86-avx2 x86-avx512
//@ [x86-avx2] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
//@ [x86-avx2] compile-flags: -C target-feature=+avx2
//@ [x86-avx2] needs-llvm-components: x86
//@ [x86-avx512] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
//@ [x86-avx512] compile-flags: -C target-feature=+avx512f,+avx512vl,+avx512bw,+avx512dq
//@ [x86-avx512] needs-llvm-components: x86
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -O

#![feature(no_core, lang_items, repr_simd, intrinsics)]
#![no_core]
#![allow(non_camel_case_types)]

// Because we don't have core yet.
#[lang = "sized"]
pub trait Sized {}

#[lang = "copy"]
trait Copy {}

#[repr(simd)]
pub struct i8x16([i8; 16]);

#[repr(simd)]
pub struct m8x16([i8; 16]);

#[repr(simd)]
pub struct f32x8([f32; 8]);

#[repr(simd)]
pub struct m32x8([i32; 8]);

#[repr(simd)]
pub struct f64x4([f64; 4]);

#[repr(simd)]
pub struct m64x4([i64; 4]);

extern "rust-intrinsic" {
    fn simd_masked_store<M, P, T>(mask: M, pointer: P, values: T);
}

// CHECK-LABEL: store_i8x16
#[no_mangle]
pub unsafe extern "C" fn store_i8x16(mask: m8x16, pointer: *mut i8, value: i8x16) {
    // Since avx2 supports no masked stores for bytes, the code tests each individual bit
    // and jumps to code that extracts individual bytes to memory.
    // x86-avx2: vpsllw xmm0, xmm0, 7
    // x86-avx2-NEXT: vpmovmskb eax, xmm0
    // x86-avx2-NEXT: test al, 1
    // x86-avx2-NEXT: jne
    // x86-avx2-NEXT: test al, 2
    // x86-avx2-NEXT: jne
    // x86-avx2-DAG: vpextrb byte ptr [rdi + 1], xmm1, 1
    // x86-avx2-DAG: vpextrb byte ptr [rdi], xmm1, 0
    //
    // x86-avx512: vpsllw xmm0, xmm0, 7
    // x86-avx512-NEXT: vpmovb2m k1, xmm0
    // x86-avx512-NEXT: vmovdqu8 xmmword ptr [rdi] {k1}, xmm1
    simd_masked_store(mask, pointer, value)
}

// CHECK-LABEL: store_f32x8
#[no_mangle]
pub unsafe extern "C" fn store_f32x8(mask: m32x8, pointer: *mut f32, value: f32x8) {
    // x86-avx2: vpslld ymm0, ymm0, 31
    // x86-avx2-NEXT: vmaskmovps ymmword ptr [rdi], ymm0, ymm1
    //
    // x86-avx512: vpslld ymm0, ymm0, 31
    // x86-avx512-NEXT: vpmovd2m k1, ymm0
    // x86-avx512-NEXT: vmovups ymmword ptr [rdi] {k1}, ymm1
    simd_masked_store(mask, pointer, value)
}

// CHECK-LABEL: store_f64x4
#[no_mangle]
pub unsafe extern "C" fn store_f64x4(mask: m64x4, pointer: *mut f64, value: f64x4) {
    // x86-avx2: vpsllq ymm0, ymm0, 63
    // x86-avx2-NEXT: vmaskmovpd ymmword ptr [rdi], ymm0, ymm1
    //
    // x86-avx512: vpsllq ymm0, ymm0, 63
    // x86-avx512-NEXT: vpmovq2m k1, ymm0
    // x86-avx512-NEXT: vmovupd ymmword ptr [rdi] {k1}, ymm1
    simd_masked_store(mask, pointer, value)
}
