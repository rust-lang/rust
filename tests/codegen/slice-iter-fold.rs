//@ compile-flags: -O
#![crate_type = "lib"]

// CHECK-LABEL: @slice_fold_to_last
// CHECK-SAME: %slice.0, [[USIZE:i[0-9]+]] noundef %slice.1)
#[no_mangle]
pub fn slice_fold_to_last(slice: &[i32]) -> Option<&i32> {
    // CHECK: %[[END:.+]] = getelementptr inbounds i32, ptr %slice.0, [[USIZE]] %slice.1
    // CHECK: %[[EMPTY:.+]] = icmp eq [[USIZE]] %slice.1, 0
    // CHECK: %[[LAST:.+]] = getelementptr i32, ptr %[[END]], i64 -1
    // CHECK: %[[R:.+]] = select i1 %[[EMPTY]], ptr null, ptr %[[LAST]]
    // CHECK: ret ptr %[[R]]
    slice.iter().fold(None, |_, i| Some(i))
}
