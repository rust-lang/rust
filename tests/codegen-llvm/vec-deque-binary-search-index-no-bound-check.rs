//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

use std::collections::VecDeque;

// Make sure no bounds check is emitted when indexing with a successful search result.

// CHECK-LABEL: @vec_deque_binary_search_index_no_bounds_check
#[no_mangle]
pub fn vec_deque_binary_search_index_no_bounds_check(deque: &VecDeque<u8>) -> u8 {
    // CHECK-NOT: expect_failed
    // CHECK-NOT: panic
    if let Ok(index) = deque.binary_search_by(|element| element.cmp(&b'\\')) {
        deque[index]
    } else {
        42
    }
}

// An unsuccessful search result is a valid insertion point.

// CHECK-LABEL: @vec_deque_binary_search_insert_no_bounds_check
#[no_mangle]
pub fn vec_deque_binary_search_insert_no_bounds_check(deque: &mut VecDeque<u8>) {
    // CHECK-NOT: panic
    if let Err(index) = deque.binary_search_by(|element| element.cmp(&b'\\')) {
        deque.insert(index, b'\\');
    }
}
