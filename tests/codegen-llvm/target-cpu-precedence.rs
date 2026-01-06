// Check that when `-Ctarget-cpu` is specified multiple times, the last
// occurrence is the value passed to LLVM.

//@ add-minicore
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ compile-flags: -Ctarget-cpu=x86-64-v2 -Ctarget-cpu=x86-64-v3
//@ compile-flags: -Cno-prepopulate-passes
//@ needs-llvm-components: x86
//@ ignore-backends: gcc

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub extern "C" fn foo() {}
// CHECK: attributes
// CHECK-SAME: "target-cpu"="x86-64-v3"
// CHECK-NOT: "target-cpu"="x86-64-v2"
