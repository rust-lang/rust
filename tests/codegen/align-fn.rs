// compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0

#![crate_type = "lib"]
#![feature(fn_align)]

// CHECK: align 16
#[no_mangle]
#[repr(align(16))]
pub fn fn_align() {}
