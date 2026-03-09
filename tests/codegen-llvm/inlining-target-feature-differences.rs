//@ only-arm
//@ ignore-backends: gcc

#![feature(arm_target_feature)]
#![crate_type = "lib"]
#![no_std]

#[inline]
fn callee_neon(a: i32, b: i32) -> i32 {
    a + b
}

#[no_mangle]
#[target_feature(enable = "neon")]
pub fn caller_neon(x: i32, y: i32) -> i32 {
    // CHECK-LABEL: define noundef i32 @caller_neon(
    // CHECK: [[R:%[0-9A-Za-z_.]+]] = add
    callee_neon(x, y)
}
