//@ revisions: x86-avx2 x86-avx512
//@ [x86-avx2] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
//@ [x86-avx2] compile-flags: -C target-feature=+avx2
//@ [x86-avx2] needs-llvm-components: x86
//@ [x86-avx512] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
//@ [x86-avx512] compile-flags: -C target-feature=+avx512f,+avx512vl,+avx512bw,+avx512dq
//@ [x86-avx512] needs-llvm-components: x86
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
impl<T: ?Sized> Copy for *const T {}

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
    fn simd_masked_load<M, P, T>(mask: M, pointer: P, values: T) -> T;
}

// CHECK-LABEL: load_i8x16
#[no_mangle]
pub unsafe extern "C" fn load_i8x16(mask: m8x16, pointer: *const i8) -> i8x16 {
    // Since avx2 supports no masked loads for bytes, the code tests each individual bit
    // and jumps to code that inserts individual bytes.
    // x86-avx2: vpsllw xmm0, xmm0, 7
    // x86-avx2-NEXT: vpmovmskb eax, xmm0
    // x86-avx2-NEXT: vpxor xmm0, xmm0
    // x86-avx2-NEXT: test al, 1
    // x86-avx2-NEXT: jne
    // x86-avx2-NEXT: test al, 2
    // x86-avx2-NEXT: jne
    // x86-avx2-DAG: movzx [[REG:[a-z]+]], byte ptr [rdi]
    // x86-avx2-NEXT: vmovd xmm0, [[REG]]
    // x86-avx2-DAG: vpinsrb xmm0, xmm0, byte ptr [rdi + 1], 1
    //
    // x86-avx512: vpsllw xmm0, xmm0, 7
    // x86-avx512-NEXT: vpmovb2m k1, xmm0
    // x86-avx512-NEXT: vmovdqu8 xmm0 {k1} {z}, xmmword ptr [rdi]
    simd_masked_load(mask, pointer, i8x16([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
}

// CHECK-LABEL: load_f32x8
#[no_mangle]
pub unsafe extern "C" fn load_f32x8(mask: m32x8, pointer: *const f32) -> f32x8 {
    // x86-avx2: vpslld ymm0, ymm0, 31
    // x86-avx2-NEXT: vmaskmovps ymm0, ymm0, ymmword ptr [rdi]
    //
    // x86-avx512: vpslld ymm0, ymm0, 31
    // x86-avx512-NEXT: vpmovd2m k1, ymm0
    // x86-avx512-NEXT: vmovups ymm0 {k1} {z}, ymmword ptr [rdi]
    simd_masked_load(mask, pointer, f32x8([0_f32, 0_f32, 0_f32, 0_f32, 0_f32, 0_f32, 0_f32, 0_f32]))
}

// CHECK-LABEL: load_f64x4
#[no_mangle]
pub unsafe extern "C" fn load_f64x4(mask: m64x4, pointer: *const f64) -> f64x4 {
    // x86-avx2: vpsllq ymm0, ymm0, 63
    // x86-avx2-NEXT: vmaskmovpd ymm0, ymm0, ymmword ptr [rdi]
    //
    // x86-avx512: vpsllq ymm0, ymm0, 63
    // x86-avx512-NEXT: vpmovq2m k1, ymm0
    // x86-avx512-NEXT: vmovupd ymm0 {k1} {z}, ymmword ptr [rdi]
    simd_masked_load(mask, pointer, f64x4([0_f64, 0_f64, 0_f64, 0_f64]))
}
