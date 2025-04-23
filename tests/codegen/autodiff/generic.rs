//@ compile-flags: -Zautodiff=Enable -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme
#![feature(autodiff)]

use std::autodiff::autodiff;

#[autodiff(d_square, Reverse, Duplicated, Active)]
fn square<T: std::ops::Mul<Output = T> + Copy>(x: &T) -> T {
    *x * *x
}

// Ensure that `d_square::<f64>` code is generated even if `square::<f64>` was never called
//
// CHECK: ; generic::square
// CHECK-NEXT: ; Function Attrs:
// CHECK-NEXT: define internal {{.*}} double
// CHECK-NEXT: start:
// CHECK-NOT: ret
// CHECK: fmul double

// Ensure that `d_square::<f32>` code is generated
//
// CHECK: ; generic::square
// CHECK-NEXT: ; Function Attrs: {{.*}}
// CHECK-NEXT: define internal {{.*}} float
// CHECK-NEXT: start:
// CHECK-NOT: ret
// CHECK: fmul float

fn main() {
    let xf32: f32 = std::hint::black_box(3.0);
    let xf64: f64 = std::hint::black_box(3.0);

    let outputf32 = square::<f32>(&xf32);
    assert_eq!(9.0, outputf32);

    let mut df_dxf64: f64 = std::hint::black_box(0.0);

    let output_f64 = d_square::<f64>(&xf64, &mut df_dxf64, 1.0);
    assert_eq!(6.0, df_dxf64);
}
