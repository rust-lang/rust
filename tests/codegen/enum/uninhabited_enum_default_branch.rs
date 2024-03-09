//@ compile-flags: -O

#![crate_type = "lib"]

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Int(u32);

const A: Int = Int(201);
const B: Int = Int(270);
const C: Int = Int(153);

// CHECK-LABEL: @foo(
// CHECK-SAME: [[TMP0:%.*]])
// CHECK-NEXT:  start:
// CHECK-NEXT:    [[TMP1:%.*]] = add i32 [[TMP0]], -201
// CHECK-NEXT:    icmp ult i32 [[TMP1]], 70
// CHECK-NEXT:    icmp eq i32 [[TMP0]], 153
// CHECK-NEXT:    [[SPEC_SELECT:%.*]] = or i1
// CHECK-NEXT:    ret i1 [[SPEC_SELECT]]
#[no_mangle]
pub fn foo(x: Int) -> bool {
    (x >= A && x <= B)
        || x == C
}
