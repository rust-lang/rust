//@ add-minicore
//@ revisions: x86-avx512
//@ [x86-avx512] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
//@ [x86-avx512] compile-flags: -C target-feature=+avx512f,+avx512vl,+avx512bw,+avx512dq
//@ [x86-avx512] needs-llvm-components: x86
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -C panic=abort

#![feature(no_core, lang_items, intrinsics)]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::simd::*;

type m64x4 = Simd<i64, 4>;
type pf64x4 = Simd<*const f64, 4>;

#[rustc_intrinsic]
unsafe fn simd_gather<V, M, P>(values: V, mask: M, pointer: P) -> V;

// CHECK-LABEL: gather_f64x4
#[no_mangle]
pub unsafe extern "C" fn gather_f64x4(mask: m64x4, ptrs: pf64x4) -> f64x4 {
    // FIXME(llvm23): This should also get checked to generate a gather instruction for avx2.
    // Currently llvm scalarizes this code, see https://github.com/llvm/llvm-project/issues/59789
    //
    // x86-avx512-NOT: vpsllq
    // x86-avx512: vpmovq2m k1, ymm0
    // x86-avx512-NEXT: vpxor xmm0, xmm0, xmm0
    // x86-avx512-NEXT: vgatherqpd ymm0 {k1}, {{(ymmword)|(qword)}} ptr [1*ymm1]
    simd_gather(f64x4::from_array([0_f64, 0_f64, 0_f64, 0_f64]), ptrs, mask)
}
