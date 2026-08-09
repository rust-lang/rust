//@ compile-flags: -Copt-level=3 -C panic=abort
#![crate_type = "lib"]
#![no_std]

// A successful search for a `char` always returns the start of a nonempty
// match, so its byte index is valid for the original string.

// Verify that the check would be visible in the generated IR.

// CHECK-LABEL: @bounds_check_is_visible
#[no_mangle]
pub fn bounds_check_is_visible(s: &str, index: usize) -> u8 {
    // CHECK: call{{.*}}panic_bounds_check
    s.as_bytes()[index]
}

// CHECK-LABEL: @find_char_index_no_bounds_check
#[no_mangle]
pub fn find_char_index_no_bounds_check(s: &str, needle: char) -> u8 {
    // CHECK-NOT: call{{.*}}panic_bounds_check
    if let Some(index) = s.find(needle) { s.as_bytes()[index] } else { 0 }
}

// CHECK-LABEL: @find_predicate_index_no_bounds_check
#[no_mangle]
pub fn find_predicate_index_no_bounds_check(s: &str, needle: char) -> u8 {
    // CHECK-NOT: call{{.*}}panic_bounds_check
    if let Some(index) = s.find(|c| c == needle) { s.as_bytes()[index] } else { 0 }
}

// CHECK-LABEL: @rfind_char_index_no_bounds_check
#[no_mangle]
pub fn rfind_char_index_no_bounds_check(s: &str, needle: char) -> u8 {
    // CHECK-NOT: call{{.*}}panic_bounds_check
    if let Some(index) = s.rfind(needle) { s.as_bytes()[index] } else { 0 }
}

// CHECK-LABEL: @rfind_predicate_index_no_bounds_check
#[no_mangle]
pub fn rfind_predicate_index_no_bounds_check(s: &str, needle: char) -> u8 {
    // CHECK-NOT: call{{.*}}panic_bounds_check
    if let Some(index) = s.rfind(|c| c == needle) { s.as_bytes()[index] } else { 0 }
}
