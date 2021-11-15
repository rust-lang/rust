// compile-flags: -O
// only-x86_64
// ignore-debug: the debug assertions in from_raw_parts get in the way

#![crate_type = "lib"]

// CHECK-LABEL: @slice_reverse_u8
#[no_mangle]
pub fn slice_reverse_u8(slice: &mut [u8]) {
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: slice_end_index_len_fail
    // CHECK: shufflevector <{{[0-9]+}} x i8>
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: slice_end_index_len_fail
    slice.reverse();
}

// CHECK-LABEL: @slice_reverse_i32
#[no_mangle]
pub fn slice_reverse_i32(slice: &mut [i32]) {
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: slice_end_index_len_fail
    // CHECK: shufflevector <{{[0-9]+}} x i32>
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: slice_end_index_len_fail
    slice.reverse();
}
