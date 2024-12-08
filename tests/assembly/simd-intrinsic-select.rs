//@ revisions: x86-avx2 x86-avx512 aarch64
//@ [x86-avx2] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
//@ [x86-avx2] compile-flags: -C target-feature=+avx2
//@ [x86-avx2] needs-llvm-components: x86
//@ [x86-avx512] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
//@ [x86-avx512] compile-flags: -C target-feature=+avx512f,+avx512vl,+avx512bw,+avx512dq
//@ [x86-avx512] needs-llvm-components: x86
//@ [aarch64] compile-flags: --target=aarch64-unknown-linux-gnu
//@ [aarch64] needs-llvm-components: aarch64
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
pub struct i8x16([i8; 16]);

#[repr(simd)]
pub struct m8x16([i8; 16]);

#[repr(simd)]
pub struct f32x4([f32; 4]);

#[repr(simd)]
pub struct m32x4([i32; 4]);

#[repr(simd)]
pub struct f64x2([f64; 2]);

#[repr(simd)]
pub struct m64x2([i64; 2]);

#[repr(simd)]
pub struct f64x4([f64; 4]);

#[repr(simd)]
pub struct m64x4([i64; 4]);

#[repr(simd)]
pub struct f64x8([f64; 8]);

#[repr(simd)]
pub struct m64x8([i64; 8]);

extern "rust-intrinsic" {
    fn simd_select<M, V>(mask: M, a: V, b: V) -> V;
}

// CHECK-LABEL: select_i8x16
#[no_mangle]
pub unsafe extern "C" fn select_i8x16(mask: m8x16, a: i8x16, b: i8x16) -> i8x16 {
    // x86-avx2: vpsllw xmm0, xmm0, 7
    // x86-avx2-NEXT: vpblendvb xmm0, xmm2, xmm1, xmm0
    //
    // x86-avx512: vpsllw xmm0, xmm0, 7
    // x86-avx512-NEXT: vpmovb2m k1, xmm0
    // x86-avx512-NEXT: vpblendmb xmm0 {k1}, xmm2, xmm1
    //
    // aarch64: shl v0.16b, v0.16b, #7
    // aarch64-NEXT: cmlt v0.16b, v0.16b, #0
    // aarch64-NEXT: bsl v0.16b, v1.16b, v2.16b
    simd_select(mask, a, b)
}

// CHECK-LABEL: select_f32x4
#[no_mangle]
pub unsafe extern "C" fn select_f32x4(mask: m32x4, a: f32x4, b: f32x4) -> f32x4 {
    // x86-avx2: vpslld xmm0, xmm0, 31
    // x86-avx2-NEXT: vblendvps xmm0, xmm2, xmm1, xmm0
    //
    // x86-avx512: vpslld xmm0, xmm0, 31
    // x86-avx512-NEXT: vpmovd2m k1, xmm0
    // x86-avx512-NEXT: vblendmps xmm0 {k1}, xmm2, xmm1
    //
    // aarch64: shl v0.4s, v0.4s, #31
    // aarch64-NEXT: cmlt v0.4s, v0.4s, #0
    // aarch64-NEXT: bsl v0.16b, v1.16b, v2.16b
    simd_select(mask, a, b)
}

// CHECK-LABEL: select_f64x2
#[no_mangle]
pub unsafe extern "C" fn select_f64x2(mask: m64x2, a: f64x2, b: f64x2) -> f64x2 {
    // x86-avx2: vpsllq xmm0, xmm0, 63
    // x86-avx2-NEXT: vblendvpd xmm0, xmm2, xmm1, xmm0
    //
    // x86-avx512: vpsllq xmm0, xmm0, 63
    // x86-avx512-NEXT: vpmovq2m k1, xmm0
    // x86-avx512-NEXT: vblendmpd xmm0 {k1}, xmm2, xmm1
    //
    // aarch64: shl v0.2d, v0.2d, #63
    // aarch64-NEXT: cmlt v0.2d, v0.2d, #0
    // aarch64-NEXT: bsl v0.16b, v1.16b, v2.16b
    simd_select(mask, a, b)
}

// CHECK-LABEL: select_f64x4
#[no_mangle]
pub unsafe extern "C" fn select_f64x4(mask: m64x4, a: f64x4, b: f64x4) -> f64x4 {
    // The parameter is a 256 bit vector which in the C abi is only valid for avx targets.
    //
    // x86-avx2: vpsllq ymm0, ymm0, 63
    // x86-avx2-NEXT: vblendvpd ymm0, ymm2, ymm1, ymm0
    //
    // x86-avx512: vpsllq ymm0, ymm0, 63
    // x86-avx512-NEXT: vpmovq2m k1, ymm0
    // x86-avx512-NEXT: vblendmpd ymm0 {k1}, ymm2, ymm1
    simd_select(mask, a, b)
}

// CHECK-LABEL: select_f64x8
#[no_mangle]
pub unsafe extern "C" fn select_f64x8(mask: m64x8, a: f64x8, b: f64x8) -> f64x8 {
    // The parameter is a 256 bit vector which in the C abi is only valid for avx512 targets.
    //
    // x86-avx512: vpsllq zmm0, zmm0, 63
    // x86-avx512-NEXT: vpmovq2m k1, zmm0
    // x86-avx512-NEXT: vblendmpd zmm0 {k1}, zmm2, zmm1
    simd_select(mask, a, b)
}
