//@ only-bpf
#![crate_type = "lib"]
#![feature(bpf_target_feature)]
#![no_std]

#[no_mangle]
#[target_feature(enable = "allows-misaligned-mem-access")]
// CHECK: define noundef zeroext i8 @foo(i8 noundef returned %arg) unnamed_addr #0 {
pub unsafe fn foo(arg: u8) -> u8 {
    arg
}
