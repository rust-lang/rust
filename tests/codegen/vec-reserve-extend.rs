//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

// CHECK-LABEL: @should_reserve_once
#[no_mangle]
pub fn should_reserve_once(v: &mut Vec<u8>) {
    // CHECK: tail call void @llvm.assume
    v.try_reserve(3).unwrap();
    // CHECK-NOT: call {{.*}}reserve
    // CHECK-NOT: call {{.*}}do_reserve_and_handle
    // CHECK-NOT: call {{.*}}__rust_alloc(
    v.extend([1, 2, 3]);
}
