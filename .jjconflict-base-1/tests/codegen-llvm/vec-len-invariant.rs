//@ compile-flags: -Copt-level=3
//@ only-64bit
//
// This test confirms that we do not reload the length of a Vec after growing it in push.

#![crate_type = "lib"]

// CHECK-LABEL: @should_load_once
#[no_mangle]
pub fn should_load_once(v: &mut Vec<u8>) {
    // CHECK: load i64
    // CHECK: call {{.*}}grow_one
    // CHECK-NOT: load i64
    // CHECK: add {{.*}}, 1
    v.push(1);
}
