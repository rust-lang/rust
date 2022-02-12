// Checks if the "fastcall" calling convention marks function arguments
// as "inreg" like the C/C++ compilers for the platforms.
// x86 only.

// compile-flags: --target i686-unknown-linux-gnu -C no-prepopulate-passes
// needs-llvm-components: x86

#![crate_type = "lib"]
#![no_core]
#![feature(no_core, lang_items)]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

pub mod tests {
    // CHECK: @f1(i32 inreg %_1, i32 inreg %_2, i32 %_3)
    #[no_mangle]
    pub extern "fastcall" fn f1(_: i32, _: i32, _: i32) {}

    // CHECK: @f2(i32* inreg %_1, i32* inreg %_2, i32* %_3)
    #[no_mangle]
    pub extern "fastcall" fn f2(_: *const i32, _: *const i32, _: *const i32) {}

    // CHECK: @f3(float %_1, i32 inreg %_2, i32 inreg %_3, i32 %_4)
    #[no_mangle]
    pub extern "fastcall" fn f3(_: f32, _: i32, _: i32, _: i32) {}

    // CHECK: @f4(i32 inreg %_1, float %_2, i32 inreg %_3, i32 %_4)
    #[no_mangle]
    pub extern "fastcall" fn f4(_: i32, _: f32, _: i32, _: i32) {}

    // CHECK: @f5(i64 %_1, i32 %_2)
    #[no_mangle]
    pub extern "fastcall" fn f5(_: i64, _: i32) {}

    // CHECK: @f6(i1 inreg noundef zeroext %_1, i32 inreg %_2, i32 %_3)
    #[no_mangle]
    pub extern "fastcall" fn f6(_: bool, _: i32, _: i32) {}
}
