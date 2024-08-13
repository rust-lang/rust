//@ only-wasm32-wasip1
//@ compile-flags: -Ctarget-feature=+relaxed-simd --crate-type=lib
//@ build-pass

use std::arch::wasm32::*;

pub fn test(a: v128, b: v128, m: v128) -> v128 {
    i64x2_relaxed_laneselect(a, b, m)
}
