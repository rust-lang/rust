//@ add-core-stubs
//@ revisions: x86-avx512
//@ [x86-avx512] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
//@ [x86-avx512] compile-flags: -C target-feature=+avx512f,+avx512vl,+avx512bw,+avx512dq
//@ [x86-avx512] needs-llvm-components: x86
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -C panic=abort

#![feature(no_core, lang_items, repr_simd, intrinsics)]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct f64x4([f64; 4]);

#[repr(simd)]
pub struct m64x4([i64; 4]);

#[repr(simd)]
pub struct pf64x4([*mut f64; 4]);

#[rustc_intrinsic]
unsafe fn simd_scatter<V, P, M>(values: V, pointer: P, mask: M);

// CHECK-LABEL: scatter_f64x4
#[no_mangle]
pub unsafe extern "C" fn scatter_f64x4(values: f64x4, ptrs: pf64x4, mask: m64x4) {
    // x86-avx512-NOT: vpsllq
    // x86-avx512: vpmovq2m k1, ymm2
    // x86-avx512-NEXT: vscatterqpd {{(ymmword)|(qword)}} ptr [1*ymm1] {k1}, ymm0
    simd_scatter(values, ptrs, mask)
}
