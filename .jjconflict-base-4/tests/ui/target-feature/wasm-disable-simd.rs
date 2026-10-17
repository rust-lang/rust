//@ only-wasm32-wasip1
//@ compile-flags: -Ctarget-feature=-simd128 --crate-type=lib
//@ build-pass

// This is a regression test of #131031.

use std::arch::wasm32::*;

#[target_feature(enable = "simd128")]
pub unsafe fn some_simd128_fn(chunk: v128) -> bool {
    u8x16_all_true(chunk)
}
