//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
//! Test for <https://github.com/rust-lang/rust/issues/108395>. Check that
//! matching on two bools with wildcards does not produce branches.
#![crate_type = "lib"]

// CHECK-LABEL: @wildcard(
#[no_mangle]
pub fn wildcard(a: u16, b: u16, v: u16) -> u16 {
    // CHECK-NOT: br
    match (a == v, b == v) {
        (true, false) => 0,
        (false, true) => u16::MAX,
        _ => 1 << 15, // half
    }
}

// CHECK-LABEL: @exhaustive(
#[no_mangle]
pub fn exhaustive(a: u16, b: u16, v: u16) -> u16 {
    // CHECK-NOT: br
    match (a == v, b == v) {
        (true, false) => 0,
        (false, true) => u16::MAX,
        (true, true) => 1 << 15,
        (false, false) => 1 << 15,
    }
}
