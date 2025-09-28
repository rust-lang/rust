//@ compile-flags: -Zautodiff=Enable,NoTT -C opt-level=3  -Clto=fat
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
#[inline(never)]
fn primal(x: f32, y: f32) -> f64 {
    (x * x * y) as f64
}

// CHECK: define internal fastcc { double, float, float } @diffeprimal(float noundef %x, float noundef %y)
// CHECK-NEXT: invertstart:
// CHECK-NEXT: %_4 = fmul float %x, %x
// CHECK-NEXT: %_3 = fmul float %_4, %y
// CHECK-NEXT: %_0 = fpext float %_3 to double
// CHECK-NEXT: %0 = fadd fast float %y, %y
// CHECK-NEXT: %1 = fmul fast float %0, %x
// CHECK-NEXT: %2 = insertvalue { double, float, float } undef, double %_0, 0
// CHECK-NEXT: %3 = insertvalue { double, float, float } %2, float %1, 1
// CHECK-NEXT: %4 = insertvalue { double, float, float } %3, float %_4, 2
// CHECK-NEXT: ret { double, float, float } %4
// CHECK-NEXT: }

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
