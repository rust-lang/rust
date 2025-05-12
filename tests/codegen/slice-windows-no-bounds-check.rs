#![crate_type = "lib"]

//@ compile-flags: -Copt-level=3

use std::slice::Windows;

// CHECK-LABEL: @naive_string_search
#[no_mangle]
pub fn naive_string_search(haystack: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    // CHECK-NOT: panic
    // CHECK-NOT: fail
    haystack.as_bytes().windows(needle.len()).position(|sub| sub == needle.as_bytes())
}

// CHECK-LABEL: @next
#[no_mangle]
pub fn next<'a>(w: &mut Windows<'a, u32>) -> Option<&'a [u32]> {
    // CHECK-NOT: panic
    // CHECK-NOT: fail
    w.next()
}

// CHECK-LABEL: @next_back
#[no_mangle]
pub fn next_back<'a>(w: &mut Windows<'a, u32>) -> Option<&'a [u32]> {
    // CHECK-NOT: panic
    // CHECK-NOT: fail
    w.next_back()
}
