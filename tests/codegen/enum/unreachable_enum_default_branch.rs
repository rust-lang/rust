//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Int(u32);

const A: Int = Int(201);
const B: Int = Int(270);
const C: Int = Int(153);

// The code is from https://github.com/rust-lang/rust/issues/119520.
// This code will basically turn into `matches!(x.partial_cmp(&A), Some(Greater | Equal))`.
// The otherwise branch must be `Less`.
// CHECK-LABEL: @implicit_match(
// CHECK-SAME: [[TMP0:%.*]])
// CHECK-NEXT:  start:
// CHECK-NEXT:    [[TMP1:%.*]] = add i32 [[TMP0]], -201
// CHECK-NEXT:    icmp ult i32 [[TMP1]], 70
// CHECK-NEXT:    icmp eq i32 [[TMP0]], 153
// CHECK-NEXT:    [[SPEC_SELECT:%.*]] = or i1
// CHECK-NEXT:    ret i1 [[SPEC_SELECT]]
#[no_mangle]
pub fn implicit_match(x: Int) -> bool {
    (x >= A && x <= B) || x == C
}

// The code is from https://github.com/rust-lang/rust/issues/110097.
// We expect it to generate the same optimized code as a full match.
// CHECK-LABEL: @if_let(
// CHECK: start:
// CHECK-NOT: zext
// CHECK: select
// CHECK-NEXT: insertvalue
// CHECK-NEXT: insertvalue
// CHECK-NEXT: ret
#[no_mangle]
pub fn if_let(val: Result<i32, ()>) -> Result<i32, ()> {
    if let Ok(x) = val { Ok(x * 2) } else { Err(()) }
}
