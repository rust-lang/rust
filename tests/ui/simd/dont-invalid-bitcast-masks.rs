//@ build-pass
//@ compile-flags: -Copt-level=3

// regression test for https://github.com/rust-lang/rust/issues/110722
// in --release we were optimizing to invalid bitcasts, due to a combination of MIR inlining and
// mostly bad repr(simd) lowering which prevented even basic splats from working
#![crate_type = "rlib"]
#![feature(portable_simd)]
use std::simd::*;
use std::simd::num::*;

pub unsafe fn mask_to_array(mask: u8) -> [i32; 8] {
    let mut output = [0; 8];
    let m = masksizex8::from_bitmask(mask as _);
    output.copy_from_slice(&m.to_int().cast::<i32>().to_array());
    output
}
