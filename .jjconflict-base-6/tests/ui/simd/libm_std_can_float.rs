//@ run-pass

// This is the converse of the other libm test.
#![feature(portable_simd)]
use std::simd::f32x4;
use std::simd::{num::SimdFloat, StdFloat};

// For SIMD float ops, the LLIR version which is used to implement the portable
// forms of them may become calls to math.h AKA libm. So, we can't guarantee
// we can compile them for #![no_std] crates.
//
// However, we can expose some of these ops via an extension trait.
fn main() {
    let x = f32x4::from_array([0.1, 0.5, 0.6, -1.5]);
    let x2 = x + x;
    let _xc = x.ceil();
    let _xf = x.floor();
    let _xr = x.round();
    let _xt = x.trunc();
    let _xfma = x.mul_add(x, x);
    let _xsqrt = x.sqrt();
    let _ = x2.abs() * x2;
}
