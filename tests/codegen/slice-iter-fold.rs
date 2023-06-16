// ignore-debug: the debug assertions get in the way
// compile-flags: -O
// min-llvm-version: 16
#![crate_type = "lib"]

// CHECK-LABEL: @slice_fold_to_last
#[no_mangle]
pub fn slice_fold_to_last(slice: &[i32]) -> Option<&i32> {
    // CHECK-NOT: loop
    // CHECK-NOT: br
    // CHECK-NOT: call
    // CHECK: ret
    slice.iter().fold(None, |_, i| Some(i))
}
