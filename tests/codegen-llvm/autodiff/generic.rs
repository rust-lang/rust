//@ compile-flags: -Zautodiff=Enable -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme
//@ revisions: F32 F64 Main

// Here we verify that the function `square` can be differentiated over f64.
// This is interesting to test, since the user never calls `square` with f64, so on it's own rustc
// would have no reason to monomorphize it that way. However, Enzyme needs the f64 version of
// `square` in order to be able to differentiate it, so we have logic to enforce the
// monomorphization. Here, we test this logic.

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[inline(never)]
fn square<T: std::ops::Mul<Output = T> + Copy>(x: &T) -> T {
    *x * *x
}

// Ensure that `d_square::<f32>` code is generated

// F32-LABEL: ; generic::square::<f32>
// F32-NEXT: ; Function Attrs: {{.*}}
// F32-NEXT: define internal {{.*}} float
// F32-NEXT: start:
// F32-NOT: ret
// F32: fmul float

// Ensure that `d_square::<f64>` code is generated even if `square::<f64>` was never called

// F64-LABEL: ; generic::d_square::<f64>
// F64-NEXT: ; Function Attrs: {{.*}}
// F64-NEXT: define internal {{.*}} void
// F64-NEXT: start:
// F64-NEXT:   {{(tail )?}}call {{(fastcc )?}}void @diffe_{{.*}}(double {{.*}}, ptr {{.*}})
// F64-NEXT: ret void

// Main-LABEL: ; generic::main
// Main: ; call generic::square::<f32>
// Main: ; call generic::d_square::<f64>

fn main() {
    let xf32: f32 = std::hint::black_box(3.0);
    let xf64: f64 = std::hint::black_box(3.0);
    let seed: f64 = std::hint::black_box(1.0);

    let outputf32 = square::<f32>(&xf32);
    assert_eq!(9.0, outputf32);

    let mut df_dxf64: f64 = std::hint::black_box(0.0);

    let output_f64 = d_square::<f64>(&xf64, &mut df_dxf64, seed);
    assert_eq!(6.0, df_dxf64);
}
