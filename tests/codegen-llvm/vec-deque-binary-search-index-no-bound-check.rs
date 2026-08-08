//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

use std::collections::VecDeque;

unsafe extern "C" {
    safe fn vec_deque_binary_search_insertion_point_out_of_bounds();
}

// Make sure no bounds check is emitted when indexing with a successful search result.

// CHECK-LABEL: @vec_deque_binary_search_index_no_bounds_check
#[no_mangle]
pub fn vec_deque_binary_search_index_no_bounds_check(deque: &VecDeque<u8>) -> u8 {
    // CHECK-NOT: expect_failed
    if let Ok(index) = deque.binary_search_by(|element| element.cmp(&b'\\')) {
        deque[index]
    } else {
        42
    }
}

// An unsuccessful search result is a valid insertion point.

// CHECK-LABEL: @vec_deque_binary_search_insertion_point_in_bounds
#[no_mangle]
pub fn vec_deque_binary_search_insertion_point_in_bounds(deque: &VecDeque<u8>) {
    // CHECK-NOT: call void @vec_deque_binary_search_insertion_point_out_of_bounds
    // CHECK: ret void
    if let Err(index) = deque.binary_search_by(|element| element.cmp(&b'\\')) {
        if index > deque.len() {
            vec_deque_binary_search_insertion_point_out_of_bounds()
        }
    }
}
