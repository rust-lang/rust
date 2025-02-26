//@ revisions: x32 x64
//@[x32] only-x86
//@[x64] only-x86_64
//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

use std::arch::global_asm;

// CHECK-LABEL: foo
// CHECK: module asm
// CHECK: module asm "{{[[:space:]]+}}jmp baz"
global_asm!(include_str!("foo.s"));

extern "C" {
    fn foo();
}

// CHECK-LABEL: @baz
#[no_mangle]
pub unsafe extern "C" fn baz() {}
