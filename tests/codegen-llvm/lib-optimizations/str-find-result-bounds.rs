//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// Make sure no bounds checks are emitted when slicing with an index returned
// by `str::find` or `str::rfind`.

// CHECK-LABEL: @find_str_prefix_no_bounds_check
#[no_mangle]
pub fn find_str_prefix_no_bounds_check<'a>(haystack: &'a str, needle: &str) -> &'a [u8] {
    // CHECK-NOT: slice_index_fail
    match haystack.find(needle) {
        Some(index) => &haystack.as_bytes()[..index],
        None => haystack.as_bytes(),
    }
}

// CHECK-LABEL: @rfind_char_suffix_no_bounds_check
#[no_mangle]
pub fn rfind_char_suffix_no_bounds_check(haystack: &str, needle: char) -> &[u8] {
    // CHECK-NOT: slice_index_fail
    match haystack.rfind(needle) {
        Some(index) => &haystack.as_bytes()[index..],
        None => haystack.as_bytes(),
    }
}

// CHECK-LABEL: @rfind_str_suffix_no_bounds_check
#[no_mangle]
pub fn rfind_str_suffix_no_bounds_check<'a>(haystack: &'a str, needle: &str) -> &'a [u8] {
    // CHECK-NOT: slice_index_fail
    match haystack.rfind(needle) {
        Some(index) => &haystack.as_bytes()[index..],
        None => haystack.as_bytes(),
    }
}
