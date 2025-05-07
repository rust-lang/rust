// Checks if the "fastcall" calling convention marks function arguments
// as "inreg" like the C/C++ compilers for the platforms.
// x86 only.

//@ add-core-stubs
//@ compile-flags: --target i686-unknown-linux-gnu -Cno-prepopulate-passes -Copt-level=3
//@ needs-llvm-components: x86

#![crate_type = "lib"]
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;
use minicore::*;

pub mod tests {
    // CHECK: @f1(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 noundef %_3)
    #[no_mangle]
    pub extern "fastcall" fn f1(_: i32, _: i32, _: i32) {}

    // CHECK: @f2(ptr inreg noundef %_1, ptr inreg noundef %_2, ptr noundef %_3)
    #[no_mangle]
    pub extern "fastcall" fn f2(_: *const i32, _: *const i32, _: *const i32) {}

    // CHECK: @f3(float noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3, i32 noundef %_4)
    #[no_mangle]
    pub extern "fastcall" fn f3(_: f32, _: i32, _: i32, _: i32) {}

    // CHECK: @f4(i32 inreg noundef %_1, float noundef %_2, i32 inreg noundef %_3, i32 noundef %_4)
    #[no_mangle]
    pub extern "fastcall" fn f4(_: i32, _: f32, _: i32, _: i32) {}

    // CHECK: @f5(i64 noundef %_1, i32 noundef %_2)
    #[no_mangle]
    pub extern "fastcall" fn f5(_: i64, _: i32) {}

    // CHECK: @f6(i1 inreg noundef zeroext %_1, i32 inreg noundef %_2, i32 noundef %_3)
    #[no_mangle]
    pub extern "fastcall" fn f6(_: bool, _: i32, _: i32) {}
}
