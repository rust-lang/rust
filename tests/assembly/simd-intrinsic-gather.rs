//@ revisions: x86-avx512
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

#[repr(simd)]
pub struct f64x4([f64; 4]);

#[repr(simd)]
pub struct m64x4([i64; 4]);

#[repr(simd)]
pub struct pf64x4([*const f64; 4]);

extern "rust-intrinsic" {
    fn simd_gather<V, M, P>(values: V, mask: M, pointer: P) -> V;
}

// CHECK-LABEL: gather_f64x4
#[no_mangle]
pub unsafe extern "C" fn gather_f64x4(mask: m64x4, ptrs: pf64x4) -> f64x4 {
    // FIXME: This should also get checked to generate a gather instruction for avx2.
    // Currently llvm scalarizes this code, see https://github.com/llvm/llvm-project/issues/59789
    //
    // x86-avx512: vpsllq ymm0, ymm0, 63
    // x86-avx512-NEXT: vpmovq2m k1, ymm0
    // x86-avx512-NEXT: vpxor xmm0, xmm0, xmm0
    // x86-avx512-NEXT: vgatherqpd ymm0 {k1}, ymmword ptr [1*ymm1]
    simd_gather(f64x4([0_f64, 0_f64, 0_f64, 0_f64]), ptrs, mask)
}
