//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 -C panic=abort
//@ compile-flags: --target armv7-unknown-linux-gnueabihf
//@ compile-flags: -C target-feature=+neon
//@ compile-flags: -Zmerge-functions=disabled
//@ needs-llvm-components: arm

#![feature(no_core, repr_simd)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct f32x4([f32; 4]);

impl Copy for f32x4 {}

macro_rules! check {
    ($func:ident $modifier:literal $reg:ident $ty:ident $mov:literal) => {
        // -Copt-level=3 and extern "C" guarantee that the selected register is always r0/s0/d0/q0
        #[no_mangle]
        pub unsafe extern "C" fn $func() -> $ty {
            let y;
            asm!(concat!($mov, " {0:", $modifier, "}, {0:", $modifier, "}"), out($reg) y);
            y
        }
    };
}

// CHECK-LABEL: reg:
// CHECK: @APP
// CHECK: mov r0, r0
// CHECK: @NO_APP
check!(reg "" reg i32 "mov");

// CHECK-LABEL: sreg:
// CHECK: @APP
// CHECK: vmov.f32 s0, s0
// CHECK: @NO_APP
check!(sreg "" sreg f32 "vmov.f32");

// CHECK-LABEL: sreg_low16:
// CHECK: @APP
// CHECK: vmov.f32 s0, s0
// CHECK: @NO_APP
check!(sreg_low16 "" sreg_low16 f32 "vmov.f32");

// CHECK-LABEL: dreg:
// CHECK: @APP
// CHECK: vmov.f64 d0, d0
// CHECK: @NO_APP
check!(dreg "" dreg f64 "vmov.f64");

// CHECK-LABEL: dreg_low16:
// CHECK: @APP
// CHECK: vmov.f64 d0, d0
// CHECK: @NO_APP
check!(dreg_low16 "" dreg_low16 f64 "vmov.f64");

// CHECK-LABEL: dreg_low8:
// CHECK: @APP
// CHECK: vmov.f64 d0, d0
// CHECK: @NO_APP
check!(dreg_low8 "" dreg_low8 f64 "vmov.f64");

// CHECK-LABEL: qreg:
// CHECK: @APP
// CHECK: vorr q0, q0, q0
// CHECK: @NO_APP
check!(qreg "" qreg f32x4 "vmov");

// CHECK-LABEL: qreg_e:
// CHECK: @APP
// CHECK: vmov.f64 d0, d0
// CHECK: @NO_APP
check!(qreg_e "e" qreg f32x4 "vmov.f64");

// CHECK-LABEL: qreg_f:
// CHECK: @APP
// CHECK: vmov.f64 d1, d1
// CHECK: @NO_APP
check!(qreg_f "f" qreg f32x4 "vmov.f64");

// CHECK-LABEL: qreg_low8:
// CHECK: @APP
// CHECK: vorr q0, q0, q0
// CHECK: @NO_APP
check!(qreg_low8 "" qreg_low8 f32x4 "vmov");

// CHECK-LABEL: qreg_low8_e:
// CHECK: @APP
// CHECK: vmov.f64 d0, d0
// CHECK: @NO_APP
check!(qreg_low8_e "e" qreg_low8 f32x4 "vmov.f64");

// CHECK-LABEL: qreg_low8_f:
// CHECK: @APP
// CHECK: vmov.f64 d1, d1
// CHECK: @NO_APP
check!(qreg_low8_f "f" qreg_low8 f32x4 "vmov.f64");

// CHECK-LABEL: qreg_low4:
// CHECK: @APP
// CHECK: vorr q0, q0, q0
// CHECK: @NO_APP
check!(qreg_low4 "" qreg_low4 f32x4 "vmov");

// CHECK-LABEL: qreg_low4_e:
// CHECK: @APP
// CHECK: vmov.f64 d0, d0
// CHECK: @NO_APP
check!(qreg_low4_e "e" qreg_low4 f32x4 "vmov.f64");

// CHECK-LABEL: qreg_low4_f:
// CHECK: @APP
// CHECK: vmov.f64 d1, d1
// CHECK: @NO_APP
check!(qreg_low4_f "f" qreg_low4 f32x4 "vmov.f64");
