//@ add-core-stubs
//@ revisions: base d32 neon
//@ assembly-output: emit-asm
//@ compile-flags: --target armv7-unknown-linux-gnueabihf
//@ compile-flags: -C opt-level=0
//@ compile-flags: -Zmerge-functions=disabled
//@[d32] compile-flags: -C target-feature=+d32
//@[neon] compile-flags: -C target-feature=+neon --cfg d32
//@[neon] filecheck-flags: --check-prefix d32
//@ needs-llvm-components: arm

#![feature(no_core, repr_simd, f16)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

extern crate minicore;
use minicore::*;

type ptr = *mut u8;

#[repr(simd)]
pub struct i8x8([i8; 8]);
#[repr(simd)]
pub struct i16x4([i16; 4]);
#[repr(simd)]
pub struct i32x2([i32; 2]);
#[repr(simd)]
pub struct i64x1([i64; 1]);
#[repr(simd)]
pub struct f16x4([f16; 4]);
#[repr(simd)]
pub struct f32x2([f32; 2]);
#[repr(simd)]
pub struct i8x16([i8; 16]);
#[repr(simd)]
pub struct i16x8([i16; 8]);
#[repr(simd)]
pub struct i32x4([i32; 4]);
#[repr(simd)]
pub struct i64x2([i64; 2]);
#[repr(simd)]
pub struct f16x8([f16; 8]);
#[repr(simd)]
pub struct f32x4([f32; 4]);

impl Copy for i8x8 {}
impl Copy for i16x4 {}
impl Copy for i32x2 {}
impl Copy for i64x1 {}
impl Copy for f16x4 {}
impl Copy for f32x2 {}
impl Copy for i8x16 {}
impl Copy for i16x8 {}
impl Copy for i32x4 {}
impl Copy for i64x2 {}
impl Copy for f16x8 {}
impl Copy for f32x4 {}

extern "C" {
    fn extern_func();
    static extern_static: u8;
}

// CHECK-LABEL: sym_fn:
// CHECK: @APP
// CHECK: bl extern_func
// CHECK: @NO_APP
#[no_mangle]
pub unsafe fn sym_fn() {
    asm!("bl {}", sym extern_func);
}

// CHECK-LABEL: sym_static:
// CHECK: @APP
// CHECK: adr r0, extern_static
// CHECK: @NO_APP
#[no_mangle]
pub unsafe fn sym_static() {
    asm!("adr r0, {}", sym extern_static);
}

// Regression test for #82052.
// CHECK-LABEL: issue_82052
// CHECK: push {{.*}}lr
// CHECK: @APP
// CHECK: @NO_APP
pub unsafe fn issue_82052() {
    asm!("", out("r14") _);
}

macro_rules! check {
    ($func:ident $ty:ident $class:ident $mov:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($mov, " {}, {}"), out($class) y, in($class) x);
            y
        }
    };
}

macro_rules! check_reg {
    ($func:ident $ty:ident $reg:tt $mov:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($mov, " ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
            y
        }
    };
}

// CHECK-LABEL: reg_i8:
// CHECK: @APP
// CHECK: mov {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: @NO_APP
check!(reg_i8 i8 reg "mov");

// CHECK-LABEL: reg_i16:
// CHECK: @APP
// CHECK: mov {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: @NO_APP
check!(reg_i16 i16 reg "mov");

// CHECK-LABEL: reg_i32:
// CHECK: @APP
// CHECK: mov {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: @NO_APP
check!(reg_i32 i32 reg "mov");

// CHECK-LABEL: reg_f16:
// CHECK: @APP
// CHECK: mov {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: @NO_APP
check!(reg_f16 f16 reg "mov");

// CHECK-LABEL: reg_f32:
// CHECK: @APP
// CHECK: mov {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: @NO_APP
check!(reg_f32 f32 reg "mov");

// CHECK-LABEL: reg_ptr:
// CHECK: @APP
// CHECK: mov {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: @NO_APP
check!(reg_ptr ptr reg "mov");

// CHECK-LABEL: sreg_i32:
// CHECK: @APP
// CHECK: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: @NO_APP
check!(sreg_i32 i32 sreg "vmov.f32");

// CHECK-LABEL: sreg_f16:
// CHECK: @APP
// CHECK: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: @NO_APP
check!(sreg_f16 f16 sreg "vmov.f32");

// CHECK-LABEL: sreg_f32:
// CHECK: @APP
// CHECK: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: @NO_APP
check!(sreg_f32 f32 sreg "vmov.f32");

// CHECK-LABEL: sreg_ptr:
// CHECK: @APP
// CHECK: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: @NO_APP
check!(sreg_ptr ptr sreg "vmov.f32");

// CHECK-LABEL: sreg_low16_i32:
// CHECK: @APP
// CHECK: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: @NO_APP
check!(sreg_low16_i32 i32 sreg_low16 "vmov.f32");

// CHECK-LABEL: sreg_low16_f16:
// CHECK: @APP
// CHECK: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: @NO_APP
check!(sreg_low16_f16 f16 sreg_low16 "vmov.f32");

// CHECK-LABEL: sreg_low16_f32:
// CHECK: @APP
// CHECK: vmov.f32 s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: @NO_APP
check!(sreg_low16_f32 f32 sreg_low16 "vmov.f32");

// d32-LABEL: dreg_i64:
// d32: @APP
// d32: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// d32: @NO_APP
#[cfg(d32)]
check!(dreg_i64 i64 dreg "vmov.f64");

// d32-LABEL: dreg_f64:
// d32: @APP
// d32: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// d32: @NO_APP
#[cfg(d32)]
check!(dreg_f64 f64 dreg "vmov.f64");

// neon-LABEL: dreg_i8x8:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_i8x8 i8x8 dreg "vmov.f64");

// neon-LABEL: dreg_i16x4:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_i16x4 i16x4 dreg "vmov.f64");

// neon-LABEL: dreg_i32x2:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_i32x2 i32x2 dreg "vmov.f64");

// neon-LABEL: dreg_i64x1:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_i64x1 i64x1 dreg "vmov.f64");

// neon-LABEL: dreg_f16x4:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_f16x4 f16x4 dreg "vmov.f64");

// neon-LABEL: dreg_f32x2:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_f32x2 f32x2 dreg "vmov.f64");

// CHECK-LABEL: dreg_low16_i64:
// CHECK: @APP
// CHECK: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// CHECK: @NO_APP
check!(dreg_low16_i64 i64 dreg_low16 "vmov.f64");

// CHECK-LABEL: dreg_low16_f64:
// CHECK: @APP
// CHECK: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// CHECK: @NO_APP
check!(dreg_low16_f64 f64 dreg_low16 "vmov.f64");

// neon-LABEL: dreg_low16_i8x8:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_low16_i8x8 i8x8 dreg_low16 "vmov.f64");

// neon-LABEL: dreg_low16_i16x4:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_low16_i16x4 i16x4 dreg_low16 "vmov.f64");

// neon-LABEL: dreg_low16_i32x2:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_low16_i32x2 i32x2 dreg_low16 "vmov.f64");

// neon-LABEL: dreg_low16_i64x1:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_low16_i64x1 i64x1 dreg_low16 "vmov.f64");

// neon-LABEL: dreg_low16_f16x4:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_low16_f16x4 f16x4 dreg_low16 "vmov.f64");

// neon-LABEL: dreg_low16_f32x2:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_low16_f32x2 f32x2 dreg_low16 "vmov.f64");

// CHECK-LABEL: dreg_low8_i64:
// CHECK: @APP
// CHECK: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// CHECK: @NO_APP
check!(dreg_low8_i64 i64 dreg_low8 "vmov.f64");

// CHECK-LABEL: dreg_low8_f64:
// CHECK: @APP
// CHECK: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// CHECK: @NO_APP
check!(dreg_low8_f64 f64 dreg_low8 "vmov.f64");

// neon-LABEL: dreg_low8_i8x8:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_low8_i8x8 i8x8 dreg_low8 "vmov.f64");

// neon-LABEL: dreg_low8_i16x4:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_low8_i16x4 i16x4 dreg_low8 "vmov.f64");

// neon-LABEL: dreg_low8_i32x2:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_low8_i32x2 i32x2 dreg_low8 "vmov.f64");

// neon-LABEL: dreg_low8_i64x1:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_low8_i64x1 i64x1 dreg_low8 "vmov.f64");

// neon-LABEL: dreg_low8_f16x4:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_low8_f16x4 f16x4 dreg_low8 "vmov.f64");

// neon-LABEL: dreg_low8_f32x2:
// neon: @APP
// neon: vmov.f64 d{{[0-9]+}}, d{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(dreg_low8_f32x2 f32x2 dreg_low8 "vmov.f64");

// neon-LABEL: qreg_i8x16:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_i8x16 i8x16 qreg "vmov");

// neon-LABEL: qreg_i16x8:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_i16x8 i16x8 qreg "vmov");

// neon-LABEL: qreg_i32x4:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_i32x4 i32x4 qreg "vmov");

// neon-LABEL: qreg_i64x2:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_i64x2 i64x2 qreg "vmov");

// neon-LABEL: qreg_f16x8:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_f16x8 f16x8 qreg "vmov");

// neon-LABEL: qreg_f32x4:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_f32x4 f32x4 qreg "vmov");

// neon-LABEL: qreg_low8_i8x16:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_low8_i8x16 i8x16 qreg_low8 "vmov");

// neon-LABEL: qreg_low8_i16x8:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_low8_i16x8 i16x8 qreg_low8 "vmov");

// neon-LABEL: qreg_low8_i32x4:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_low8_i32x4 i32x4 qreg_low8 "vmov");

// neon-LABEL: qreg_low8_i64x2:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_low8_i64x2 i64x2 qreg_low8 "vmov");

// neon-LABEL: qreg_low8_f16x8:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_low8_f16x8 f16x8 qreg_low8 "vmov");

// neon-LABEL: qreg_low8_f32x4:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_low8_f32x4 f32x4 qreg_low8 "vmov");

// neon-LABEL: qreg_low4_i8x16:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_low4_i8x16 i8x16 qreg_low4 "vmov");

// neon-LABEL: qreg_low4_i16x8:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_low4_i16x8 i16x8 qreg_low4 "vmov");

// neon-LABEL: qreg_low4_i32x4:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_low4_i32x4 i32x4 qreg_low4 "vmov");

// neon-LABEL: qreg_low4_i64x2:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_low4_i64x2 i64x2 qreg_low4 "vmov");

// neon-LABEL: qreg_low4_f16x8:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_low4_f16x8 f16x8 qreg_low4 "vmov");

// neon-LABEL: qreg_low4_f32x4:
// neon: @APP
// neon: vorr q{{[0-9]+}}, q{{[0-9]+}}, q{{[0-9]+}}
// neon: @NO_APP
#[cfg(neon)]
check!(qreg_low4_f32x4 f32x4 qreg_low4 "vmov");

// CHECK-LABEL: r0_i8:
// CHECK: @APP
// CHECK: mov r0, r0
// CHECK: @NO_APP
check_reg!(r0_i8 i8 "r0" "mov");

// CHECK-LABEL: r0_i16:
// CHECK: @APP
// CHECK: mov r0, r0
// CHECK: @NO_APP
check_reg!(r0_i16 i16 "r0" "mov");

// CHECK-LABEL: r0_i32:
// CHECK: @APP
// CHECK: mov r0, r0
// CHECK: @NO_APP
check_reg!(r0_i32 i32 "r0" "mov");

// CHECK-LABEL: r0_f16:
// CHECK: @APP
// CHECK: mov r0, r0
// CHECK: @NO_APP
check_reg!(r0_f16 f16 "r0" "mov");

// CHECK-LABEL: r0_f32:
// CHECK: @APP
// CHECK: mov r0, r0
// CHECK: @NO_APP
check_reg!(r0_f32 f32 "r0" "mov");

// CHECK-LABEL: r0_ptr:
// CHECK: @APP
// CHECK: mov r0, r0
// CHECK: @NO_APP
check_reg!(r0_ptr ptr "r0" "mov");

// CHECK-LABEL: s0_i32:
// CHECK: @APP
// CHECK: vmov.f32 s0, s0
// CHECK: @NO_APP
check_reg!(s0_i32 i32 "s0" "vmov.f32");

// CHECK-LABEL: s0_f16:
// CHECK: @APP
// CHECK: vmov.f32 s0, s0
// CHECK: @NO_APP
check_reg!(s0_f16 f16 "s0" "vmov.f32");

// CHECK-LABEL: s0_f32:
// CHECK: @APP
// CHECK: vmov.f32 s0, s0
// CHECK: @NO_APP
check_reg!(s0_f32 f32 "s0" "vmov.f32");

// CHECK-LABEL: s0_ptr:
// CHECK: @APP
// CHECK: vmov.f32 s0, s0
// CHECK: @NO_APP
check_reg!(s0_ptr ptr "s0" "vmov.f32");

// FIXME(#126797): "d0" should work with `i64` and `f64` even when `d32` is disabled.
// d32-LABEL: d0_i64:
// d32: @APP
// d32: vmov.f64 d0, d0
// d32: @NO_APP
#[cfg(d32)]
check_reg!(d0_i64 i64 "d0" "vmov.f64");

// d32-LABEL: d0_f64:
// d32: @APP
// d32: vmov.f64 d0, d0
// d32: @NO_APP
#[cfg(d32)]
check_reg!(d0_f64 f64 "d0" "vmov.f64");

// neon-LABEL: d0_i8x8:
// neon: @APP
// neon: vmov.f64 d0, d0
// neon: @NO_APP
#[cfg(neon)]
check_reg!(d0_i8x8 i8x8 "d0" "vmov.f64");

// neon-LABEL: d0_i16x4:
// neon: @APP
// neon: vmov.f64 d0, d0
// neon: @NO_APP
#[cfg(neon)]
check_reg!(d0_i16x4 i16x4 "d0" "vmov.f64");

// neon-LABEL: d0_i32x2:
// neon: @APP
// neon: vmov.f64 d0, d0
// neon: @NO_APP
#[cfg(neon)]
check_reg!(d0_i32x2 i32x2 "d0" "vmov.f64");

// neon-LABEL: d0_i64x1:
// neon: @APP
// neon: vmov.f64 d0, d0
// neon: @NO_APP
#[cfg(neon)]
check_reg!(d0_i64x1 i64x1 "d0" "vmov.f64");

// neon-LABEL: d0_f16x4:
// neon: @APP
// neon: vmov.f64 d0, d0
// neon: @NO_APP
#[cfg(neon)]
check_reg!(d0_f16x4 f16x4 "d0" "vmov.f64");

// neon-LABEL: d0_f32x2:
// neon: @APP
// neon: vmov.f64 d0, d0
// neon: @NO_APP
#[cfg(neon)]
check_reg!(d0_f32x2 f32x2 "d0" "vmov.f64");

// neon-LABEL: q0_i8x16:
// neon: @APP
// neon: vorr q0, q0, q0
// neon: @NO_APP
#[cfg(neon)]
check_reg!(q0_i8x16 i8x16 "q0" "vmov");

// neon-LABEL: q0_i16x8:
// neon: @APP
// neon: vorr q0, q0, q0
// neon: @NO_APP
#[cfg(neon)]
check_reg!(q0_i16x8 i16x8 "q0" "vmov");

// neon-LABEL: q0_i32x4:
// neon: @APP
// neon: vorr q0, q0, q0
// neon: @NO_APP
#[cfg(neon)]
check_reg!(q0_i32x4 i32x4 "q0" "vmov");

// neon-LABEL: q0_i64x2:
// neon: @APP
// neon: vorr q0, q0, q0
// neon: @NO_APP
#[cfg(neon)]
check_reg!(q0_i64x2 i64x2 "q0" "vmov");

// neon-LABEL: q0_f16x8:
// neon: @APP
// neon: vorr q0, q0, q0
// neon: @NO_APP
#[cfg(neon)]
check_reg!(q0_f16x8 f16x8 "q0" "vmov");

// neon-LABEL: q0_f32x4:
// neon: @APP
// neon: vorr q0, q0, q0
// neon: @NO_APP
#[cfg(neon)]
check_reg!(q0_f32x4 f32x4 "q0" "vmov");
