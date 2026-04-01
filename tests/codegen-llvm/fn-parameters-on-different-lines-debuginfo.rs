//! Make sure that line debuginfo of function parameters are correct even if
//! they are not on the same line. Regression test for
// <https://github.com/rust-lang/rust/issues/45010>.

//@ compile-flags: -g -Copt-level=0

#[rustfmt::skip] // Having parameters on different lines is crucial for this test.
pub fn foo(
    x_parameter_not_in_std: i32,
    y_parameter_not_in_std: i32,
) -> i32 {
    x_parameter_not_in_std + y_parameter_not_in_std
}

fn main() {
    foo(42, 43); // Ensure `wasm32-wasip1` keeps `foo()` (even if `-Copt-level=0`)
}

// CHECK: !DILocalVariable(name: "x_parameter_not_in_std", arg: 1,
// CHECK-SAME: line: 9
// CHECK: !DILocalVariable(name: "y_parameter_not_in_std", arg: 2,
// CHECK-SAME: line: 10
