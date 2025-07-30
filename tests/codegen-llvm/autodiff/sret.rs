//@ compile-flags: -Zautodiff=Enable -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

// This test is almost identical to the scalar.rs one,
// but we intentionally add a few more floats.
// `df` would ret `{ f64, f32, f32 }`, but is lowered as an sret.
// We therefore use this test to verify some of our sret handling.

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[no_mangle]
#[autodiff_reverse(df, Active, Active, Active)]
fn primal(x: f32, y: f32) -> f64 {
    (x * x * y) as f64
}

// CHECK: %_4.i = fmul float %x, %x
// CHECK-NEXT: %_3.i = fmul float %_4.i, %y
// CHECK-NEXT: %_0.i = fpext float %_3.i to double
// CHECK-NEXT: %3 = fadd fast float %y, %y
// CHECK-NEXT: %4 = fmul fast float %3, %x
// CHECK-NEXT: store double %_0.i, ptr %r1, align 8
// CHECK-NEXT: store float %4, ptr %r2, align 4
// CHECK-NEXT: store float %_4.i, ptr %r3, align 4
fn main() {
    let x = std::hint::black_box(3.0);
    let y = std::hint::black_box(2.5);
    let scalar = std::hint::black_box(1.0);
    let (r1, r2, r3) = df(x, y, scalar);
    // 3*3*2.5 = 22.5
    assert_eq!(r1, 22.5);
    // 2*x*y = 2*3*2.5 = 15.0
    assert_eq!(r2, 15.0);
    // x*x*1 = 3*3 = 9
    assert_eq!(r3, 9.0);
}
