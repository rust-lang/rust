// ignore-debug: the debug assertions get in the way
// compile-flags: -O
#![crate_type = "lib"]

// CHECK-LABEL: @slice_for_each_to_last =
// CHECK-SAME: alias{{.*}}@slice_fold_to_last

// CHECK-LABEL: @slice_fold_to_last(
#[no_mangle]
pub fn slice_fold_to_last(slice: &[i32]) -> Option<&i32> {
    // CHECK-NOT: loop
    // CHECK-NOT: br
    // CHECK-NOT: call
    // CHECK: ret
    slice.iter().fold(None, |_, i| Some(i))
}

#[no_mangle]
pub fn slice_for_each_to_last(slice: &[i32]) -> Option<&i32> {
    let mut last = None;
    slice.iter().for_each(|i| last = Some(i));
    last
}

// CHECK-LABEL: @slice_rfold_to_first(
#[no_mangle]
pub fn slice_rfold_to_first(slice: &[i32]) -> Option<&i32> {
    // CHECK-NOT: loop
    // CHECK-NOT: br
    // CHECK-NOT: call
    // CHECK: ret
    slice.iter().rfold(None, |_, i| Some(i))
}
