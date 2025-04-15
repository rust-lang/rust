//@ compile-flags: -Zautodiff=Enable -Zautodiff=NoPostopt -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

#![feature(autodiff)]

use std::autodiff::autodiff;

#[autodiff(d_square, Reverse, Duplicated, Active)]
fn square(x: &f64) -> f64 {
    x * x
}
// CHECK: ; Function Attrs: alwaysinline noinline
// CHECK-NEXT: declare double @__enzyme_autodiff_ZN6inline8d_square17h021c74e92c259cdeE(...) local_unnamed_addr #8
fn main() {
    let x = std::hint::black_box(3.0);
    let mut dx1 = std::hint::black_box(1.0);
    let _ = d_square(&x, &mut dx1, 1.0);
    assert_eq!(dx1, 6.0);
}
