//@ add-minicore
//@ compile-flags: --target mips64el-unknown-linux-gnuabi64 -Copt-level=3
//@ needs-llvm-components: mips

#![feature(no_core)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct ReturnedAsFloat(f32, ());
#[repr(C)]
pub struct NotReturnedAsFloat(f32, [u8; 0]);

// CHECK-LABEL: define inreg float @foo(
#[unsafe(no_mangle)]
pub extern "C" fn foo(a: ReturnedAsFloat) -> ReturnedAsFloat {
    a
}

// CHECK-LABEL: define{{.*}} i64 @bar(
#[unsafe(no_mangle)]
pub extern "C" fn bar(b: NotReturnedAsFloat) -> NotReturnedAsFloat {
    b
}
