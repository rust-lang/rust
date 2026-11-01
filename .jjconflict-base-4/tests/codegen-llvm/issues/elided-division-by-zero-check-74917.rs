// Tests that there is no check for dividing by zero since the
// denominator, `(x - y)`, will always be greater than 0 since `x > y`.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @issue_74917
#[no_mangle]
pub fn issue_74917(x: u16, y: u16) -> u16 {
    // CHECK-NOT: panic
    if x > y { 100 / (x - y) } else { 100 }
}
