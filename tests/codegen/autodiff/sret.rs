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

// CHECK:define internal fastcc void @_ZN4sret2df17h93be4316dd8ea006E(ptr dead_on_unwind noalias nocapture noundef nonnull writable writeonly align 8 dereferenceable(16) initializes((0, 16)) %_0, float noundef %x, float noundef %y)
// CHECK-NEXT:start:
// CHECK-NEXT:  %0 = tail call fastcc { double, float, float } @diffeprimal(float %x, float %y)
// CHECK-NEXT:  %.elt = extractvalue { double, float, float } %0, 0
// CHECK-NEXT:  store double %.elt, ptr %_0, align 8
// CHECK-NEXT:  %_0.repack1 = getelementptr inbounds nuw i8, ptr %_0, i64 8
// CHECK-NEXT:  %.elt2 = extractvalue { double, float, float } %0, 1
// CHECK-NEXT:  store float %.elt2, ptr %_0.repack1, align 8
// CHECK-NEXT:  %_0.repack3 = getelementptr inbounds nuw i8, ptr %_0, i64 12
// CHECK-NEXT:  %.elt4 = extractvalue { double, float, float } %0, 2
// CHECK-NEXT:  store float %.elt4, ptr %_0.repack3, align 4
// CHECK-NEXT:  ret void
// CHECK-NEXT:}

fn main() {
    let x = std::hint::black_box(3.0);
    let y = std::hint::black_box(2.5);
    let scalar = std::hint::black_box(1.0);
    let (r1, r2, r3) = df(x, y, scalar);
    // 3*3*1.5 = 22.5
    assert_eq!(r1, 22.5);
    // 2*x*y = 2*3*2.5 = 15.0
    assert_eq!(r2, 15.0);
    // x*x*1 = 3*3 = 9
    assert_eq!(r3, 9.0);
}
