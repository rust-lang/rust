//@ compile-flags: -Copt-level=3
//@ only-x86_64
//@ ignore-std-debug-assertions (debug assertions prevent generating shufflevector)

#![crate_type = "lib"]

// CHECK-LABEL: @slice_reverse_u8
#[no_mangle]
pub fn slice_reverse_u8(slice: &mut [u8]) {
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: slice_index_fail
    // CHECK: shufflevector <{{[0-9]+}} x i8>
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: slice_index_fail
    slice.reverse();
}

// CHECK-LABEL: @slice_reverse_i32
#[no_mangle]
pub fn slice_reverse_i32(slice: &mut [i32]) {
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: slice_index_fail
    // CHECK: shufflevector <{{[0-9]+}} x i32>
    // CHECK-NOT: panic_bounds_check
    // CHECK-NOT: slice_index_fail
    slice.reverse();
}
