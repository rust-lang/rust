//@ compile-flags: -Zautodiff=Enable -C opt-level=3 -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

// Test that basic autodiff still works with our TypeTree infrastructure
#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_simple, Duplicated, Active)]
#[no_mangle]
#[inline(never)]
fn simple(x: &f64) -> f64 {
    2.0 * x
}

// CHECK-LABEL: @simple
// CHECK: fmul double

// The derivative function should be generated normally
// CHECK-LABEL: diffesimple
// CHECK: fadd fast double

fn main() {
    let x = std::hint::black_box(3.0);
    let output = simple(&x);
    assert_eq!(6.0, output);

    let mut df_dx = 0.0;
    let output_ = d_simple(&x, &mut df_dx, 1.0);
    assert_eq!(output, output_);
    assert_eq!(2.0, df_dx);
}
