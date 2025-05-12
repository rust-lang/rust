//@ compile-flags: -Copt-level=3
//@ ignore-std-debug-assertions
// (with debug assertions turned on, `assert_unchecked` generates a real assertion)

#![crate_type = "lib"]
#![feature(try_with_capacity)]

// CHECK-LABEL: @with_capacity_does_not_grow1
#[no_mangle]
pub fn with_capacity_does_not_grow1() -> Vec<u32> {
    let v = Vec::with_capacity(1234);
    // CHECK: call {{.*}}__rust_alloc(
    // CHECK-NOT: call {{.*}}__rust_realloc
    // CHECK-NOT: call {{.*}}capacity_overflow
    // CHECK-NOT: call {{.*}}finish_grow
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: memcpy
    // CHECK-NOT: memset
    v
}

// CHECK-LABEL: @try_with_capacity_does_not_grow2
#[no_mangle]
pub fn try_with_capacity_does_not_grow2() -> Option<Vec<Vec<u8>>> {
    let v = Vec::try_with_capacity(1234).ok()?;
    // CHECK: call {{.*}}__rust_alloc(
    // CHECK-NOT: call {{.*}}__rust_realloc
    // CHECK-NOT: call {{.*}}capacity_overflow
    // CHECK-NOT: call {{.*}}finish_grow
    // CHECK-NOT: call {{.*}}handle_alloc_error
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: memcpy
    // CHECK-NOT: memset
    Some(v)
}
