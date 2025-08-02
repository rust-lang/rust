//@ compile-flags: -Zautodiff=Enable -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme
#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn square(x: &f64) -> f64 {
    x * x
}

// square
// CHECK: %_0.i = fmul double %_2.i, %_2.i

// d_square
// CHECK: %0 = fadd fast double %_2.i, %_2.i
fn main() {
    let x = std::hint::black_box(3.0);
    let output = square(&x);
    assert_eq!(9.0, output);

    let mut df_dx = 0.0;
    let output_ = d_square(&x, &mut df_dx, 1.0);
    assert_eq!(output, output_);
    assert_eq!(6.0, df_dx);
}
