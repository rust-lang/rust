//@ compile-flags: -Zautodiff=Enable -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme
#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
fn square<T: std::ops::Mul<Output = T> + Copy>(x: &T) -> T {
    *x * *x
}

// Ensure that `square::<f32>` code is generated
//
// CHECK: %1 = fmul float %xf32, %xf32

// Ensure that `d_square::<f64>` code is generated even if `square::<f64>` was never called
//
// CHECK: define internal { double } @diffe_ZN7generic6square17he5c855620985cd59E

fn main() {
    let xf32: f32 = std::hint::black_box(3.0);
    let xf64: f64 = std::hint::black_box(3.0);

    let outputf32 = square::<f32>(&xf32);
    assert_eq!(9.0, outputf32);

    let mut df_dxf64: f64 = std::hint::black_box(0.0);

    let output_f64 = d_square::<f64>(&xf64, &mut df_dxf64, 1.0);
    assert_eq!(6.0, df_dxf64);
}
