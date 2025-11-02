//@ min-llvm-version: 21
//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
//! Tests that matching + eq on `Option<FieldlessEnum>` produces a simple compare with no branching

#![crate_type = "lib"]

#[derive(PartialEq)]
pub enum TwoNum {
    A,
    B,
}

#[derive(PartialEq)]
pub enum ThreeNum {
    A,
    B,
    C,
}

// CHECK-LABEL: @match_two
#[no_mangle]
pub fn match_two(a: Option<TwoNum>, b: Option<TwoNum>) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: icmp eq i8
    // CHECK-NEXT: ret
    match (a, b) {
        (Some(x), Some(y)) => x == y,
        (Some(_), None) => false,
        (None, Some(_)) => false,
        (None, None) => true,
    }
}

// CHECK-LABEL: @match_three
#[no_mangle]
pub fn match_three(a: Option<ThreeNum>, b: Option<ThreeNum>) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: icmp eq
    // CHECK-NEXT: ret
    match (a, b) {
        (Some(x), Some(y)) => x == y,
        (Some(_), None) => false,
        (None, Some(_)) => false,
        (None, None) => true,
    }
}

// CHECK-LABEL: @match_two_ref
#[no_mangle]
pub fn match_two_ref(a: &Option<TwoNum>, b: &Option<TwoNum>) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: load i8
    // CHECK-NEXT: load i8
    // CHECK-NEXT: icmp eq i8
    // CHECK-NEXT: ret
    match (a, b) {
        (Some(x), Some(y)) => x == y,
        (Some(_), None) => false,
        (None, Some(_)) => false,
        (None, None) => true,
    }
}

// CHECK-LABEL: @match_three_ref
#[no_mangle]
pub fn match_three_ref(a: &Option<ThreeNum>, b: &Option<ThreeNum>) -> bool {
    // CHECK-NEXT: start:
    // CHECK-NEXT: load i8
    // CHECK-NEXT: load i8
    // CHECK-NEXT: icmp eq
    // CHECK-NEXT: ret
    match (a, b) {
        (Some(x), Some(y)) => x == y,
        (Some(_), None) => false,
        (None, Some(_)) => false,
        (None, None) => true,
    }
}
