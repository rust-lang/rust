//@ compile-flags: -O
#![crate_type = "lib"]

// check that slice.len() has range metadata

// CHECK-LABEL: @slice_len
#[no_mangle]
pub fn slice_len(slice: &&[i32]) -> usize {
    // CHECK: load {{i64|i32}}
    // CHECK-SAME: !range
    slice.len()
    // CHECK: ret
}
