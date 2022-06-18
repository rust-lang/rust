// compile-flags: -O
// only-x86_64
// ignore-debug

#![crate_type = "lib"]

// CHECK-LABEL: @vec_zero_scalar
#[no_mangle]
pub fn vec_zero_scalar(n: usize) -> Vec<i32> {
    // CHECK-NOT: __rust_alloc(
    // CHECK: __rust_alloc_zeroed(
    // CHECK-NOT: __rust_alloc(
    vec![0; n]
}

// CHECK-LABEL: @vec_zero_rgb48
#[no_mangle]
pub fn vec_zero_rgb48(n: usize) -> Vec<[u16; 3]> {
    // CHECK-NOT: __rust_alloc(
    // CHECK: __rust_alloc_zeroed(
    // CHECK-NOT: __rust_alloc(
    vec![[0, 0, 0]; n]
}

// CHECK-LABEL: @vec_zero_array_32
#[no_mangle]
pub fn vec_zero_array_32(n: usize) -> Vec<[i64; 32]> {
    // CHECK-NOT: __rust_alloc(
    // CHECK: __rust_alloc_zeroed(
    // CHECK-NOT: __rust_alloc(
    vec![[0_i64; 32]; n]
}
