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
#[inline(never)]
fn square(x: &f64) -> f64 {
    x * x
}

#[autodiff_reverse(d_square2, Duplicated, Active)]
#[inline(never)]
fn square2(x: &f64) -> f64 {
    x * x
}

// CHECK:; identical_fnc::main
// CHECK-NEXT:; Function Attrs:
// CHECK-NEXT:define internal void @_ZN13identical_fnc4main17h6009e4f751bf9407E()
// CHECK-NEXT:start:
// CHECK-NOT:br
// CHECK-NOT:ret
// CHECK:; call identical_fnc::d_square
// CHECK-NEXT:call fastcc void @_ZN13identical_fnc8d_square[[HASH:.+]](double %x.val, ptr noalias noundef align 8 dereferenceable(8) %dx1)
// CHECK:; call identical_fnc::d_square
// CHECK-NEXT:call fastcc void @_ZN13identical_fnc8d_square[[HASH]](double %x.val, ptr noalias noundef align 8 dereferenceable(8) %dx2)

fn main() {
    let x = std::hint::black_box(3.0);
    let mut dx1 = std::hint::black_box(1.0);
    let mut dx2 = std::hint::black_box(1.0);
    let _ = d_square(&x, &mut dx1, 1.0);
    let _ = d_square2(&x, &mut dx2, 1.0);
    assert_eq!(dx1, 6.0);
    assert_eq!(dx2, 6.0);
}
