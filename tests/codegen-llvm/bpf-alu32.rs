//@ only-bpf
#![crate_type = "lib"]
#![feature(bpf_target_feature)]
#![no_std]

#[no_mangle]
#[target_feature(enable = "alu32")]
// CHECK: define i8 @foo(i8 returned %arg) unnamed_addr #0 {
pub unsafe fn foo(arg: u8) -> u8 {
    arg
}
