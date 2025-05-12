//@ build-pass
//@ compile-flags: -Copt-level=3
//@ only-x86_64
// ignore-tidy-linelength

// regression test for https://github.com/rust-lang/rust/issues/110707
// in --release we were optimizing to invalid bitcasts, due to a combination of MIR inlining and
// mostly bad repr(simd) lowering which prevented even basic splats from working

#![crate_type = "rlib"]
#![feature(portable_simd)]
use std::simd::*;
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn fast_round_sse(i: f32x8) -> f32x8 {
    let a = i.to_array();
    let [low, high]: [[f32; 4]; 2] =
        unsafe { std::mem::transmute::<[f32; 8], [[f32; 4]; 2]>(a) };

    let low = f32x4::from(_mm_round_ps::<{_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC}>(f32x4::from_array(low).into()));
    let high = f32x4::from(_mm_round_ps::<{_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC}>(f32x4::from_array(high).into()));

    let a: [f32; 8] =
        unsafe { std::mem::transmute::<[[f32; 4]; 2], [f32; 8]>([low.to_array(), high.to_array()]) };
    f32x8::from_array(a)
}
