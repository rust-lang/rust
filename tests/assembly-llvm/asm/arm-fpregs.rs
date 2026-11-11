//@ add-minicore
//@ revisions: armv7r armv8r
//@ assembly-output: emit-asm
//@[armv7r] compile-flags: --target armv7r-none-eabihf
//@[armv8r] compile-flags: --target armv8r-none-eabihf
//@ needs-llvm-components: arm

#![feature(f16, no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: sp:
// CHECK: vmov.f32 s{{([0-9]|1[0-5])}}, s{{([0-9]|1[0-5])}}
#[no_mangle]
pub unsafe fn sp(x: f32) -> f32 {
    let y;
    asm!("vmov.f32 {}, {}", out(sreg_low16) y, in(sreg_low16) x);
    y
}

// CHECK-LABEL: hp:
// CHECK: vmov.f32 s{{([0-9]|1[0-5])}}, s{{([0-9]|1[0-5])}}
#[no_mangle]
pub unsafe fn hp(x: f16) -> f16 {
    let y;
    asm!("vmov.f32 {}, {}", out(sreg_low16) y, in(sreg_low16) x);
    y
}

// Use vmov dZ, rX, rY so this works on single-precision-only targets,
// where vmov.f64 is not available.

// CHECK-LABEL: d16:
// CHECK: vmov d{{([0-9]|1[0-5])}}, r{{([0-9]|1[0-5])}}, r{{([0-9]|1[0-5])}}
#[no_mangle]
pub unsafe fn d16(a: i32, b: i32) -> f64 {
    let x;
    asm!("vmov {}, {}, {}", out(dreg_low16) x, in(reg) a, in(reg) b);
    x
}

// CHECK-LABEL: d8:
// CHECK: vmov d{{[0-7]}}, r{{([0-9]|1[0-5])}}, r{{([0-9]|1[0-5])}}
#[no_mangle]
pub unsafe fn d8(a: i32, b: i32) -> f64 {
    let x;
    asm!("vmov {}, {}, {}", out(dreg_low8) x, in(reg) a, in(reg) b);
    x
}
