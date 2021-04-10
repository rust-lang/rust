// ignore-debug: the debug assertions get in the way
// compile-flags: -O
// min-llvm-version: 11.0
#![crate_type = "lib"]

// Ensure that trivial casts and allocation recycling of vec elements are O(1) by containing no
// loops, function calls or backwards branches

// CHECK-LABEL: @vec_iterator_cast
#[no_mangle]
pub fn vec_iterator_cast(vec: Vec<isize>) -> Vec<usize> {
    // CHECK-NOT: loop
    // CHECK-NOT: call
    // CHECK-NOT: br
    // CHECK: ret void
    vec.into_iter().map(|e| e as usize).collect()
}

// CHECK-LABEL: @vec_iterator_storage_reuse
#[no_mangle]
pub fn vec_iterator_storage_reuse(vec: Vec<u32>) -> Vec<char> {
    // CHECK-NOT: loop
    // CHECK-NOT: call
    // CHECK-NOT: br
    // CHECK: ret void
    vec.into_iter().filter_map(|e| None).collect()
}
