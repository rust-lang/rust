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

// CHECK:define internal fastcc double @diffesquare(double %x.0.val, ptr nocapture nonnull align 8 %"x'"
// CHECK-NEXT:invertstart:
// CHECK-NEXT:  %_0 = fmul double %x.0.val, %x.0.val
// CHECK-NEXT:  %0 = fadd fast double %x.0.val, %x.0.val
// CHECK-NEXT:  %1 = load double, ptr %"x'", align 8
// CHECK-NEXT:  %2 = fadd fast double %1, %0
// CHECK-NEXT:  store double %2, ptr %"x'", align 8
// CHECK-NEXT:  ret double %_0
// CHECK-NEXT:}

fn main() {
    let x = std::hint::black_box(3.0);
    let output = square(&x);
    assert_eq!(9.0, output);

    let mut df_dx = 0.0;
    let output_ = d_square(&x, &mut df_dx, 1.0);
    assert_eq!(output, output_);
    assert_eq!(6.0, df_dx);
}
