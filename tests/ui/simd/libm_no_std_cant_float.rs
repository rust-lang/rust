#![crate_type = "rlib"]
#![no_std]
#![feature(portable_simd)]
use core::simd::f32x4;
use core::simd::num::SimdFloat;

// For SIMD float ops, the LLIR version which is used to implement the portable
// forms of them may become calls to math.h AKA libm. So, we can't guarantee
// we can compile them for #![no_std] crates.
// Someday we may solve this.
// Until then, this test at least guarantees these functions require std.
fn guarantee_no_std_nolibm_calls() -> f32x4 {
    let x = f32x4::from_array([0.1, 0.5, 0.6, -1.5]);
    let x2 = x + x;
    let _xc = x.ceil(); //~ ERROR E0599
    let _xf = x.floor(); //~ ERROR E0599
    let _xr = x.round(); //~ ERROR E0599
    let _xt = x.trunc(); //~ ERROR E0599
    let _xfma = x.mul_add(x, x); //~ ERROR E0599
    let _xsqrt = x.sqrt(); //~ ERROR E0599
    x2.abs() * x2
}
