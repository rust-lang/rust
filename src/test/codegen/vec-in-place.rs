// ignore-debug: the debug assertions get in the way
// compile-flags: -O
#![crate_type = "lib"]

// Ensure that trivial casts of vec elements are O(1)

// CHECK-LABEL: @vec_iterator_cast
#[no_mangle]
pub fn vec_iterator_cast(vec: Vec<isize>) -> Vec<usize> {
    // CHECK-NOT: loop
    // CHECK-NOT: call
    vec.into_iter().map(|e| e as usize).collect()
}
