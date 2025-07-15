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
#![feature(core_intrinsics)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
fn square(x: &f64) -> f64 {
    x * x
}

#[autodiff_reverse(d_square2, Duplicated, Active)]
fn square2(x: &f64) -> f64 {
    x * x
}

// CHECK:; identical_fnc::main
// CHECK-NEXT:; Function Attrs:
// CHECK-NEXT:define internal void @_ZN13identical_fnc4main17hf4dbc69c8d2f9130E()
// CHECK-NEXT:start:
// CHECK-NOT:br
// CHECK-NOT:ret
// CHECK:call fastcc void @diffe_ZN13identical_fnc6square17hdfa1c645848284b7E(double %x.val, ptr %dx1)
// CHECK-NEXT:call fastcc void @diffe_ZN13identical_fnc6square17hdfa1c645848284b7E(double %x.val, ptr %dx2)

fn main() {
    let x = std::hint::black_box(3.0);
    let mut dx1 = std::hint::black_box(1.0);
    let mut dx2 = std::hint::black_box(1.0);
    let _ = d_square(&x, &mut dx1, 1.0);
    let _ = d_square2(&x, &mut dx2, 1.0);
    assert_eq!(dx1, 6.0);
    assert_eq!(dx2, 6.0);
}
