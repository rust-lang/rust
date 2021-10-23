// compile-flags: -O
// ignore-debug: the debug assertions get in the way
#![crate_type = "lib"]

// Make sure no bounds checks are emitted when slicing or indexing
// with an index from `binary_search`.

// CHECK-LABEL: @binary_search_index_no_bounds_check
#[no_mangle]
pub fn binary_search_index_no_bounds_check(s: &[u8]) -> u8 {
    // CHECK-NOT: panic
    // CHECK-NOT: slice_index_len_fail
    if let Ok(idx) = s.binary_search(&b'\\') {
        s[idx]
    } else {
        42
    }
}
