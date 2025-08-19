// Checks how `regparm` flag works with different calling conventions:
// marks function arguments as "inreg" like the C/C++ compilers for the platforms.
// x86 only.

//@ add-core-stubs
//@ compile-flags: --target i686-unknown-linux-gnu -Cno-prepopulate-passes -Copt-level=3 -Ctarget-feature=+avx
//@ needs-llvm-components: x86

//@ revisions:regparm0 regparm1 regparm2 regparm3
//@[regparm0] compile-flags: -Zregparm=0
//@[regparm1] compile-flags: -Zregparm=1
//@[regparm2] compile-flags: -Zregparm=2
//@[regparm3] compile-flags: -Zregparm=3

#![crate_type = "lib"]
#![no_core]
#![feature(no_core, lang_items, repr_simd)]

extern crate minicore;
use minicore::*;

pub mod tests {
    // regparm doesn't work for "fastcall" calling conv (only 2 inregs)
    // CHECK: @f1(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 noundef %_3)
    #[no_mangle]
    pub extern "fastcall" fn f1(_: i32, _: i32, _: i32) {}

    // regparm0: @f3(i32 noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regparm1: @f3(i32 inreg noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regparm2: @f3(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 noundef %_3)
    // regparm3: @f3(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    #[no_mangle]
    pub extern "C" fn f3(_: i32, _: i32, _: i32) {}

    // regparm0: @f4(i32 noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regparm1: @f4(i32 inreg noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regparm2: @f4(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 noundef %_3)
    // regparm3: @f4(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    #[no_mangle]
    pub extern "cdecl" fn f4(_: i32, _: i32, _: i32) {}

    // regparm0: @f5(i32 noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regparm1: @f5(i32 inreg noundef %_1, i32 noundef %_2, i32 noundef %_3)
    // regparm2: @f5(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 noundef %_3)
    // regparm3: @f5(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3)
    #[no_mangle]
    pub extern "stdcall" fn f5(_: i32, _: i32, _: i32) {}

    // regparm doesn't work for thiscall
    // CHECK: @f6(i32 noundef %_1, i32 noundef %_2, i32 noundef %_3)
    #[no_mangle]
    pub extern "thiscall" fn f6(_: i32, _: i32, _: i32) {}

    struct S1 {
        x1: i32,
    }
    // regparm0: @f7(i32 noundef %_1, i32 noundef %_2, i32 noundef %_3, i32 noundef %_4)
    // regparm1: @f7(i32 inreg noundef %_1, i32 noundef %_2, i32 noundef %_3, i32 noundef %_4)
    // regparm2: @f7(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 noundef %_3, i32 noundef %_4)
    // regparm3: @f7(i32 inreg noundef %_1, i32 inreg noundef %_2, i32 inreg noundef %_3,
    // regparm3-SAME: i32 noundef %_4)
    #[no_mangle]
    pub extern "C" fn f7(_: i32, _: i32, _: S1, _: i32) {}

    #[repr(C)]
    struct S2 {
        x1: i32,
        x2: i32,
    }
    // regparm0: @f8(i32 noundef %_1, i32 noundef %_2, ptr {{.*}} %_3, i32 noundef %_4)
    // regparm1: @f8(i32 inreg noundef %_1, i32 noundef %_2, ptr {{.*}} %_3, i32 noundef %_4)
    // regparm2: @f8(i32 inreg noundef %_1, i32 inreg noundef %_2, ptr {{.*}} %_3, i32 noundef %_4)
    // regparm3: @f8(i32 inreg noundef %_1, i32 inreg noundef %_2, ptr {{.*}} %_3,
    // regparm3-SAME: i32 inreg noundef %_4)
    #[no_mangle]
    pub extern "C" fn f8(_: i32, _: i32, _: S2, _: i32) {}

    // regparm0: @f9(i1 noundef zeroext %_1, i16 noundef signext %_2, i64 noundef %_3,
    // regparm0-SAME: i128 noundef %_4)
    // regparm1: @f9(i1 inreg noundef zeroext %_1, i16 noundef signext %_2, i64 noundef %_3,
    // regparm1-SAME: i128 noundef %_4)
    // regparm2: @f9(i1 inreg noundef zeroext %_1, i16 inreg noundef signext %_2, i64 noundef %_3,
    // regparm2-SAME: i128 noundef %_4)
    // regparm3: @f9(i1 inreg noundef zeroext %_1, i16 inreg noundef signext %_2, i64 noundef %_3,
    // regparm3-SAME: i128 noundef %_4)
    #[no_mangle]
    pub extern "C" fn f9(_: bool, _: i16, _: i64, _: u128) {}

    // regparm0: @f10(float noundef %_1, double noundef %_2, i1 noundef zeroext %_3,
    // regparm0-SAME: i16 noundef signext %_4)
    // regparm1: @f10(float noundef %_1, double noundef %_2, i1 inreg noundef zeroext %_3,
    // regparm1-SAME: i16 noundef signext %_4)
    // regparm2: @f10(float noundef %_1, double noundef %_2, i1 inreg noundef zeroext %_3,
    // regparm2-SAME: i16 inreg noundef signext %_4)
    // regparm3: @f10(float noundef %_1, double noundef %_2, i1 inreg noundef zeroext %_3,
    // regparm3-SAME: i16 inreg noundef signext %_4)
    #[no_mangle]
    pub extern "C" fn f10(_: f32, _: f64, _: bool, _: i16) {}

    #[allow(non_camel_case_types)]
    #[repr(simd)]
    pub struct __m128([f32; 4]);

    // regparm0: @f11(i32 noundef %_1, <4 x float> %_2, i32 noundef %_3, i32 noundef %_4)
    // regparm1: @f11(i32 inreg noundef %_1, <4 x float> %_2, i32 noundef %_3, i32 noundef %_4)
    // regparm2: @f11(i32 inreg noundef %_1, <4 x float> %_2, i32 inreg noundef %_3,
    // regparm2-SAME: i32 noundef %_4)
    // regparm3: @f11(i32 inreg noundef %_1, <4 x float> %_2, i32 inreg noundef %_3,
    // regparm3-SAME: i32 inreg noundef %_4)
    #[no_mangle]
    pub extern "C" fn f11(_: i32, _: __m128, _: i32, _: i32) {}

    #[allow(non_camel_case_types)]
    #[repr(simd)]
    pub struct __m256([f32; 8]);

    // regparm0: @f12(i32 noundef %_1, <8 x float> %_2, i32 noundef %_3, i32 noundef %_4)
    // regparm1: @f12(i32 inreg noundef %_1, <8 x float> %_2, i32 noundef %_3, i32 noundef %_4)
    // regparm2: @f12(i32 inreg noundef %_1, <8 x float> %_2, i32 inreg noundef %_3,
    // regparm2-SAME: i32 noundef %_4)
    // regparm3: @f12(i32 inreg noundef %_1, <8 x float> %_2, i32 inreg noundef %_3,
    // regparm3-SAME: i32 inreg noundef %_4)
    #[no_mangle]
    pub extern "C" fn f12(_: i32, _: __m256, _: i32, _: i32) {}
}
