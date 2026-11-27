//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib
//@ only-x86_64
//@ ignore-sgx

#![feature(portable_simd)]

use std::arch::asm;
use std::simd::Simd;

#[target_feature(enable = "avx")]
#[no_mangle]
// CHECK-LABEL: convert:
pub unsafe fn convert(a: *const f32) -> Simd<f32, 8> {
    // CHECK: vbroadcastss (%{{[er][a-ds0-9][xpi0-9]?}}), {{%ymm[0-7]}}
    let b: Simd<f32, 8>;
    unsafe {
        asm!(
            "vbroadcastss {b}, [{a}]",
            a = in(reg) a,
            b = out(ymm_reg) b,
        );
    }
    b
}
