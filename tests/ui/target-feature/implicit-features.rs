//@ only-wasm32-wasip1
//@ compile-flags: --crate-type=lib
//@ build-pass

use std::arch::wasm32::*;

#[target_feature(enable = "relaxed-simd")]
pub fn test(a: v128, b: v128, m: v128) -> v128 {
    i64x2_relaxed_laneselect(a, b, m)
}
