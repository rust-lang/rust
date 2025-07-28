//@ add-core-stubs
//@ revisions: linux apple win
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes

//@[linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[linux] needs-llvm-components: x86
//@[apple] compile-flags: --target x86_64-apple-darwin
//@[apple] needs-llvm-components: x86
//@[win] compile-flags: --target x86_64-pc-windows-msvc
//@[win] needs-llvm-components: x86

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct Rgb8 {
    r: u8,
    g: u8,
    b: u8,
}

#[repr(transparent)]
pub struct Rgb8Wrap(Rgb8);

// CHECK: i24 @test_Rgb8Wrap(i24{{( %0)?}})
#[no_mangle]
pub extern "sysv64" fn test_Rgb8Wrap(_: Rgb8Wrap) -> Rgb8Wrap {
    loop {}
}

#[repr(C)]
pub union FloatBits {
    float: f32,
    bits: u32,
}

#[repr(transparent)]
pub struct SmallUnion(FloatBits);

// CHECK: i32 @test_SmallUnion(i32{{( %0)?}})
#[no_mangle]
pub extern "sysv64" fn test_SmallUnion(_: SmallUnion) -> SmallUnion {
    loop {}
}
