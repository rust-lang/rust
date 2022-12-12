// This is used to ICE with MIR inlining enabled due to an invalid bitcast.
// run-pass
// compile-flags: -O -Zmir-opt-level=3
#![feature(portable_simd)]

use std::simd::Simd;

fn main() {
    let a = Simd::from_array([0, 4, 1, 5]);
    let b = Simd::from_array([2, 6, 3, 7]);
    let (x, y) = a.deinterleave(b);
    assert_eq!(x.to_array(), [0, 1, 2, 3]);
    assert_eq!(y.to_array(), [4, 5, 6, 7]);
}
