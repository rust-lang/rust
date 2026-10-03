//@ add-minicore
//@ compile-flags: --target=bpfel-unknown-none -C opt-level=0
//@ needs-llvm-components: bpf
#![crate_type = "lib"]
#![feature(bpf_target_feature, no_core)]
#![no_core]
#![no_std]

extern crate minicore;
use minicore::*;

#[no_mangle]
#[target_feature(enable = "alu32")]
// CHECK: define {{.*}}i8 @foo(i8 {{.*}}%arg) unnamed_addr #0
// CHECK: attributes #0 = { {{.*}}"target-features"="{{[^"]*}}+alu32{{.*}} }
pub unsafe fn foo(arg: u8) -> u8 {
    arg
}
