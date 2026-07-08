//@ add-minicore
//@ revisions: loongarch32 loongarch64

//@ assembly-output: emit-asm
//@ needs-llvm-components: loongarch

//@[loongarch32] compile-flags: --target loongarch32-unknown-none
//@[loongarch32] compile-flags: -Ctarget-feature=+32s,+lasx

//@[loongarch64] compile-flags: --target loongarch64-unknown-none
//@[loongarch64] compile-flags: -Ctarget-feature=+lasx

//@ compile-flags: -Ctarget-feature=+32s
//@ compile-flags: -Zmerge-functions=disabled

#![feature(asm_experimental_reg, no_core, repr_simd)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct i8x16([i8; 16]);
#[repr(simd)]
pub struct i16x8([i16; 8]);
#[repr(simd)]
pub struct i32x4([i32; 4]);
#[repr(simd)]
pub struct i64x2([i64; 2]);
#[repr(simd)]
pub struct f32x4([f32; 4]);
#[repr(simd)]
pub struct f64x2([f64; 2]);
#[repr(simd)]
pub struct i8x32([i8; 32]);
#[repr(simd)]
pub struct i16x16([i16; 16]);
#[repr(simd)]
pub struct i32x8([i32; 8]);
#[repr(simd)]
pub struct i64x4([i64; 4]);
#[repr(simd)]
pub struct f32x8([f32; 8]);
#[repr(simd)]
pub struct f64x4([f64; 4]);

impl Copy for i8x16 {}
impl Copy for i16x8 {}
impl Copy for i32x4 {}
impl Copy for i64x2 {}
impl Copy for f32x4 {}
impl Copy for f64x2 {}
impl Copy for i8x32 {}
impl Copy for i16x16 {}
impl Copy for i32x8 {}
impl Copy for i64x4 {}
impl Copy for f32x8 {}
impl Copy for f64x4 {}

macro_rules! check { ($func:ident, $ty:ty, $class:ident, $code:literal) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        let y;
        asm!($code, out($class) y, in($class) x);
        y
    }
};}

// CHECK-LABEL: freg_f32_lsx:
// CHECK: #APP
// CHECK: vfadd.s $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(freg_f32_lsx, f32, freg, "vfadd.s {1:w}, {0:w}, {0:w}");

// CHECK-LABEL: freg_f64_lasx:
// CHECK: #APP
// CHECK: xvfadd.d $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(freg_f64_lasx, f64, freg, "xvfadd.d {1:u}, {0:u}, {0:u}");

// CHECK-LABEL: vreg_i8x16_lsx:
// CHECK: #APP
// CHECK: vadd.b $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(vreg_i8x16_lsx, i8x16, vreg, "vadd.b {1}, {0}, {0}");

// CHECK-LABEL: vreg_i8x16_lasx:
// CHECK: #APP
// CHECK: xvadd.b $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(vreg_i8x16_lasx, i8x16, vreg, "xvadd.b {1:u}, {0:u}, {0:u}");

// CHECK-LABEL: vreg_i16x8_lsx:
// CHECK: #APP
// CHECK: vadd.h $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(vreg_i16x8_lsx, i16x8, vreg, "vadd.h {1}, {0}, {0}");

// CHECK-LABEL: vreg_i16x8_lasx:
// CHECK: #APP
// CHECK: xvadd.h $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(vreg_i16x8_lasx, i16x8, vreg, "xvadd.h {1:u}, {0:u}, {0:u}");

// CHECK-LABEL: vreg_i32x4_lsx:
// CHECK: #APP
// CHECK: vadd.w $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(vreg_i32x4_lsx, i32x4, vreg, "vadd.w {1}, {0}, {0}");

// CHECK-LABEL: vreg_i32x4_lasx:
// CHECK: #APP
// CHECK: xvadd.w $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(vreg_i32x4_lasx, i32x4, vreg, "xvadd.w {1:u}, {0:u}, {0:u}");

// CHECK-LABEL: vreg_i64x2_lsx:
// CHECK: #APP
// CHECK: vadd.d $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(vreg_i64x2_lsx, i64x2, vreg, "vadd.d {1}, {0}, {0}");

// CHECK-LABEL: vreg_i64x2_lasx:
// CHECK: #APP
// CHECK: xvadd.d $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(vreg_i64x2_lasx, i64x2, vreg, "xvadd.d {1:u}, {0:u}, {0:u}");

// CHECK-LABEL: vreg_f32x4_lsx:
// CHECK: #APP
// CHECK: vfadd.s $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(vreg_f32x4_lsx, f32x4, vreg, "vfadd.s {1}, {0}, {0}");

// CHECK-LABEL: vreg_f32x4_lasx:
// CHECK: #APP
// CHECK: xvfadd.s $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(vreg_f32x4_lasx, f32x4, vreg, "xvfadd.s {1:u}, {0:u}, {0:u}");

// CHECK-LABEL: vreg_f64x2_lsx:
// CHECK: #APP
// CHECK: vfadd.d $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(vreg_f64x2_lsx, f64x2, vreg, "vfadd.d {1}, {0}, {0}");

// CHECK-LABEL: vreg_f64x2_lasx:
// CHECK: #APP
// CHECK: xvfadd.d $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(vreg_f64x2_lasx, f64x2, vreg, "xvfadd.d {1:u}, {0:u}, {0:u}");

// CHECK-LABEL: xreg_i8x32_lasx:
// CHECK: #APP
// CHECK: xvadd.b $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(xreg_i8x32_lasx, i8x32, xreg, "xvadd.b {1}, {0}, {0}");

// CHECK-LABEL: xreg_i8x32_lsx:
// CHECK: #APP
// CHECK: vadd.b $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(xreg_i8x32_lsx, i8x32, xreg, "vadd.b {1:w}, {0:w}, {0:w}");

// CHECK-LABEL: xreg_i16x16_lasx:
// CHECK: #APP
// CHECK: xvadd.h $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(xreg_i16x16_lasx, i16x16, xreg, "xvadd.h {1}, {0}, {0}");

// CHECK-LABEL: xreg_i16x16_lsx:
// CHECK: #APP
// CHECK: vadd.h $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(xreg_i16x16_lsx, i16x16, xreg, "vadd.h {1:w}, {0:w}, {0:w}");

// CHECK-LABEL: xreg_i32x8_lasx:
// CHECK: #APP
// CHECK: xvadd.w $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(xreg_i32x8_lasx, i32x8, xreg, "xvadd.w {1}, {0}, {0}");

// CHECK-LABEL: xreg_i32x8_lsx:
// CHECK: #APP
// CHECK: vadd.w $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(xreg_i32x8_lsx, i32x8, xreg, "vadd.w {1:w}, {0:w}, {0:w}");

// CHECK-LABEL: xreg_i64x4_lasx:
// CHECK: #APP
// CHECK: xvadd.d $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(xreg_i64x4_lasx, i64x4, xreg, "xvadd.d {1}, {0}, {0}");

// CHECK-LABEL: xreg_i64x4_lsx:
// CHECK: #APP
// CHECK: vadd.d $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(xreg_i64x4_lsx, i64x4, xreg, "vadd.d {1:w}, {0:w}, {0:w}");

// CHECK-LABEL: xreg_f32x8_lasx:
// CHECK: #APP
// CHECK: xvfadd.s $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(xreg_f32x8_lasx, f32x8, xreg, "xvfadd.s {1}, {0}, {0}");

// CHECK-LABEL: xreg_f32x8_lsx:
// CHECK: #APP
// CHECK: vfadd.s $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(xreg_f32x8_lsx, f32x8, xreg, "vfadd.s {1:w}, {0:w}, {0:w}");

// CHECK-LABEL: xreg_f64x4_lasx:
// CHECK: #APP
// CHECK: xvfadd.d $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}, $xr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(xreg_f64x4_lasx, f64x4, xreg, "xvfadd.d {1}, {0}, {0}");

// CHECK-LABEL: xreg_f64x4_lsx:
// CHECK: #APP
// CHECK: vfadd.d $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}, $vr{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(xreg_f64x4_lsx, f64x4, xreg, "vfadd.d {1:w}, {0:w}, {0:w}");
