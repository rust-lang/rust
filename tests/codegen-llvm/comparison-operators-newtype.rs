// The `derive(PartialOrd)` for a newtype doesn't override `lt`/`le`/`gt`/`ge`.
// This double-checks that the `Option<Ordering>` intermediate values used
// in the operators for such a type all optimize away.

//@ compile-flags: -C opt-level=1

#![crate_type = "lib"]

use std::cmp::Ordering;

#[derive(PartialOrd, PartialEq)]
pub struct Foo(u16);

// CHECK-LABEL: @check_lt
// CHECK-SAME: (i16{{.*}} %[[A:.+]], i16{{.*}} %[[B:.+]])
#[no_mangle]
pub fn check_lt(a: Foo, b: Foo) -> bool {
    // CHECK: %[[R:.+]] = icmp ult i16 %[[A]], %[[B]]
    // CHECK-NEXT: ret i1 %[[R]]
    a < b
}

// CHECK-LABEL: @check_le
// CHECK-SAME: (i16{{.*}} %[[A:.+]], i16{{.*}} %[[B:.+]])
#[no_mangle]
pub fn check_le(a: Foo, b: Foo) -> bool {
    // CHECK: %[[R:.+]] = icmp ule i16 %[[A]], %[[B]]
    // CHECK-NEXT: ret i1 %[[R]]
    a <= b
}

// CHECK-LABEL: @check_gt
// CHECK-SAME: (i16{{.*}} %[[A:.+]], i16{{.*}} %[[B:.+]])
#[no_mangle]
pub fn check_gt(a: Foo, b: Foo) -> bool {
    // CHECK: %[[R:.+]] = icmp ugt i16 %[[A]], %[[B]]
    // CHECK-NEXT: ret i1 %[[R]]
    a > b
}

// CHECK-LABEL: @check_ge
// CHECK-SAME: (i16{{.*}} %[[A:.+]], i16{{.*}} %[[B:.+]])
#[no_mangle]
pub fn check_ge(a: Foo, b: Foo) -> bool {
    // CHECK: %[[R:.+]] = icmp uge i16 %[[A]], %[[B]]
    // CHECK-NEXT: ret i1 %[[R]]
    a >= b
}
