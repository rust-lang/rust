//@ compile-flags: -Zautodiff=Enable -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme
//
// Each autodiff invocation creates a new placeholder function, which we will replace on llvm-ir
// level. If a user tries to differentiate two identical functions within the same compilation unit,
// then LLVM might merge them in release mode before AD. In that case we can't rewrite one of the
// merged placeholder function anymore, and compilation would fail. We prevent this by disabling
// LLVM's merge_function pass before AD. Here we implicetely test that our solution keeps working.
// We also explicetly test that we keep running merge_function after AD, by checking for two
// identical function calls in the LLVM-IR, while having two different calls in the Rust code.
#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
fn square(x: &f64) -> f64 {
    x * x
}

#[autodiff_reverse(d_square2, Duplicated, Active)]
fn square2(x: &f64) -> f64 {
    x * x
}

// CHECK: %0 = fadd fast double %x.val, %x.val
// CHECK-NEXT: %1 = load double, ptr %dx1, align 8
// CHECK-NEXT: %2 = fadd fast double %1, %0
// CHECK-NEXT: store double %2, ptr %dx1, align 8
// CHECK-NEXT: %3 = load double, ptr %dx2, align 8
// CHECK-NEXT: %4 = fadd fast double %3, %0
// CHECK-NEXT: store double %4, ptr %dx2, align 8
fn main() {
    let x = std::hint::black_box(3.0);
    let mut dx1 = std::hint::black_box(1.0);
    let mut dx2 = std::hint::black_box(1.0);
    let _ = d_square(&x, &mut dx1, 1.0);
    let _ = d_square2(&x, &mut dx2, 1.0);
    assert_eq!(dx1, 6.0);
    assert_eq!(dx2, 6.0);
}
