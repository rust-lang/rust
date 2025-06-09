//@ compile-flags: -Zautodiff=Enable -C opt-level=3  -Clto=fat -Zautodiff=NoPostopt
//@ no-prefer-dynamic
//@ needs-enzyme

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
fn square(x: &f64) -> f64 {
    x * x
}

// CHECK: ; inline::d_square
// CHECK-NEXT: ; Function Attrs: alwaysinline
// CHECK-NOT: noinline
// CHECK-NEXT: define internal fastcc void @_ZN6inline8d_square17h021c74e92c259cdeE
fn main() {
    let x = std::hint::black_box(3.0);
    let mut dx1 = std::hint::black_box(1.0);
    let _ = d_square(&x, &mut dx1, 1.0);
    assert_eq!(dx1, 6.0);
}
