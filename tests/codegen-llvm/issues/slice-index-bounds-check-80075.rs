// Tests that no bounds check panic is generated for `j` since
// `j <= i < data.len()`.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @issue_80075
#[no_mangle]
pub fn issue_80075(data: &[u8], i: usize, j: usize) -> u8 {
    // CHECK-NOT: panic_bounds_check
    if i < data.len() && j <= i { data[j] } else { 0 }
}
