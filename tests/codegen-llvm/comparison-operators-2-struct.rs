//@ compile-flags: -C opt-level=1

// The `derive(PartialOrd)` for a 2-field type doesn't override `lt`/`le`/`gt`/`ge`.
// This double-checks that the `Option<Ordering>` intermediate values used
// in the operators for such a type all optimize away.

#![crate_type = "lib"]

use std::cmp::Ordering;

#[derive(PartialOrd, PartialEq)]
pub struct Foo(i32, u32);

// CHECK-LABEL: @check_lt(
// CHECK-SAME: i32{{.+}}%[[A0:.+]], i32{{.+}}%[[A1:.+]], i32{{.+}}%[[B0:.+]], i32{{.+}}%[[B1:.+]])
#[no_mangle]
pub fn check_lt(a: Foo, b: Foo) -> bool {
    // CHECK-DAG: %[[EQ:.+]] = icmp eq i32 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[R0:.+]] = icmp slt i32 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[R1:.+]] = icmp ult i32 %[[A1]], %[[B1]]
    // CHECK: %[[R:.+]] = select i1 %[[EQ]], i1 %[[R1]], i1 %[[R0]]
    // CHECK-NEXT: ret i1 %[[R]]
    a < b
}

// CHECK-LABEL: @check_le(
// CHECK-SAME: i32{{.+}}%[[A0:.+]], i32{{.+}}%[[A1:.+]], i32{{.+}}%[[B0:.+]], i32{{.+}}%[[B1:.+]])
#[no_mangle]
pub fn check_le(a: Foo, b: Foo) -> bool {
    // CHECK-DAG: %[[EQ:.+]] = icmp eq i32 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[R0:.+]] = icmp sle i32 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[R1:.+]] = icmp ule i32 %[[A1]], %[[B1]]
    // CHECK: %[[R:.+]] = select i1 %[[EQ]], i1 %[[R1]], i1 %[[R0]]
    // CHECK-NEXT: ret i1 %[[R]]
    a <= b
}

// CHECK-LABEL: @check_gt(
// CHECK-SAME: i32{{.+}}%[[A0:.+]], i32{{.+}}%[[A1:.+]], i32{{.+}}%[[B0:.+]], i32{{.+}}%[[B1:.+]])
#[no_mangle]
pub fn check_gt(a: Foo, b: Foo) -> bool {
    // CHECK-DAG: %[[EQ:.+]] = icmp eq i32 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[R0:.+]] = icmp sgt i32 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[R1:.+]] = icmp ugt i32 %[[A1]], %[[B1]]
    // CHECK: %[[R:.+]] = select i1 %[[EQ]], i1 %[[R1]], i1 %[[R0]]
    // CHECK-NEXT: ret i1 %[[R]]
    a > b
}

// CHECK-LABEL: @check_ge(
// CHECK-SAME: i32{{.+}}%[[A0:.+]], i32{{.+}}%[[A1:.+]], i32{{.+}}%[[B0:.+]], i32{{.+}}%[[B1:.+]])
#[no_mangle]
pub fn check_ge(a: Foo, b: Foo) -> bool {
    // CHECK-DAG: %[[EQ:.+]] = icmp eq i32 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[R0:.+]] = icmp sge i32 %[[A0]], %[[B0]]
    // CHECK-DAG: %[[R1:.+]] = icmp uge i32 %[[A1]], %[[B1]]
    // CHECK: %[[R:.+]] = select i1 %[[EQ]], i1 %[[R1]], i1 %[[R0]]
    // CHECK-NEXT: ret i1 %[[R]]
    a >= b
}
