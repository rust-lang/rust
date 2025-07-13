//! Make sure that line debuginfo of function parameters are correct even if
//! they are not on the same line. Regression test for
// <https://github.com/rust-lang/rust/issues/45010>.

//@ compile-flags: -g -Copt-level=0

#![crate_type = "dylib"]
#[rustfmt::skip] // Having parameters on different lines is crucial for this test.
pub fn foo(
    x: i32,
    y: i32)
    -> i32
{ x + y }

// CHECK: !DILocalVariable(name: "x", arg: 1,
// CHECK-SAME: line: 10
// CHECK: !DILocalVariable(name: "y", arg: 2,
// CHECK-SAME: line: 11
