//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// Ensure that the simple case of rotating by a constant 1 optimizes to the obvious thing

// CHECK-LABEL: @rotate_left_by_one
#[no_mangle]
pub fn rotate_left_by_one(slice: &mut [i32]) {
    // CHECK-NOT: phi
    // CHECK-NOT: call
    // CHECK-NOT: load
    // CHECK-NOT: store
    // CHECK-NOT: getelementptr
    // CHECK: %[[END:.+]] = getelementptr
    // CHECK-NEXT: %[[DIM:.+]] = getelementptr
    // CHECK-NEXT: %[[LAST:.+]] = load
    // CHECK-NEXT: %[[FIRST:.+]] = shl
    // CHECK-NEXT: call void @llvm.memmove
    // CHECK-NEXT: store i32 %[[LAST]], ptr %[[DIM:.+]]
    // CHECK-NOT: phi
    // CHECK-NOT: call
    // CHECK-NOT: load
    // CHECK-NOT: store
    // CHECK-NOT: getelementptr
    // CHECK: ret void
    if !slice.is_empty() {
        slice.rotate_left(1);
    }
}
