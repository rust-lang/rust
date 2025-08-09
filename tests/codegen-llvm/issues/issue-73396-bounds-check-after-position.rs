//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

// Make sure no bounds checks are emitted when slicing or indexing
// with an index from `position()` or `rposition()`.

// CHECK-LABEL: @position_slice_to_no_bounds_check
#[no_mangle]
pub fn position_slice_to_no_bounds_check(s: &[u8]) -> &[u8] {
    // CHECK-NOT: panic
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: unreachable
    if let Some(idx) = s.iter().position(|b| *b == b'\\') { &s[..idx] } else { s }
}

// CHECK-LABEL: @position_slice_from_no_bounds_check
#[no_mangle]
pub fn position_slice_from_no_bounds_check(s: &[u8]) -> &[u8] {
    // CHECK-NOT: panic
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: unreachable
    if let Some(idx) = s.iter().position(|b| *b == b'\\') { &s[idx..] } else { s }
}

// CHECK-LABEL: @position_index_no_bounds_check
#[no_mangle]
pub fn position_index_no_bounds_check(s: &[u8]) -> u8 {
    // CHECK-NOT: panic
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: unreachable
    if let Some(idx) = s.iter().position(|b| *b == b'\\') { s[idx] } else { 42 }
}
// CHECK-LABEL: @rposition_slice_to_no_bounds_check
#[no_mangle]
pub fn rposition_slice_to_no_bounds_check(s: &[u8]) -> &[u8] {
    // CHECK-NOT: panic
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: unreachable
    if let Some(idx) = s.iter().rposition(|b| *b == b'\\') { &s[..idx] } else { s }
}

// CHECK-LABEL: @rposition_slice_from_no_bounds_check
#[no_mangle]
pub fn rposition_slice_from_no_bounds_check(s: &[u8]) -> &[u8] {
    // CHECK-NOT: panic
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: unreachable
    if let Some(idx) = s.iter().rposition(|b| *b == b'\\') { &s[idx..] } else { s }
}

// CHECK-LABEL: @rposition_index_no_bounds_check
#[no_mangle]
pub fn rposition_index_no_bounds_check(s: &[u8]) -> u8 {
    // CHECK-NOT: panic
    // CHECK-NOT: slice_index_fail
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: unreachable
    if let Some(idx) = s.iter().rposition(|b| *b == b'\\') { s[idx] } else { 42 }
}
