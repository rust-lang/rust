//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: --target riscv64gc-unknown-linux-gnu -C target-feature=+v
//@ compile-flags: -Zmerge-functions=disabled
//@ needs-llvm-components: riscv
//@ ignore-backends: gcc

#![feature(asm_experimental_reg, no_core, repr_simd)]
#![crate_type = "rlib"]
#![no_core]

extern crate minicore;
use minicore::simd::*;
use minicore::*;

macro_rules! check {
    ($func:ident $ty:ident) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!("vmv.v.v {0}, {1}", out(vreg) y, in(vreg) x);
            y
        }
    };
}

// CHECK-LABEL: vreg_i8x16:
// CHECK: #APP
// CHECK: vmv.v.v v{{[0-9]+}}, v{{[0-9]+}}
// CHECK: #NO_APP
check!(vreg_i8x16 i8x16);

// CHECK-LABEL: vreg_i16x8:
// CHECK: #APP
// CHECK: vmv.v.v v{{[0-9]+}}, v{{[0-9]+}}
// CHECK: #NO_APP
check!(vreg_i16x8 i16x8);

// CHECK-LABEL: vreg_i32x4:
// CHECK: #APP
// CHECK: vmv.v.v v{{[0-9]+}}, v{{[0-9]+}}
// CHECK: #NO_APP
check!(vreg_i32x4 i32x4);

// CHECK-LABEL: vreg_i64x2:
// CHECK: #APP
// CHECK: vmv.v.v v{{[0-9]+}}, v{{[0-9]+}}
// CHECK: #NO_APP
check!(vreg_i64x2 i64x2);

check!(vreg_i8x32 i8x32);
check!(vreg_i16x16 i16x16);
check!(vreg_i32x8 i32x8);
check!(vreg_i64x4 i64x4);

#[repr(simd)]
struct V128([i8; 128]);
impl Copy for V128 {}

check!(vreg_i8x128 V128);

// CHECK-LABEL: mixed_regs:
// CHECK: #APP
// CHECK: vmv.v.v v0, v{{[0-9]+}}
// CHECK: vmv.v.v v{{[0-9]+}}, v0
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn mixed_regs(x: i8x16) -> i8x16 {
    let m: i8x16;
    let y: i8x16;
    asm!("vmv.v.v v0, {0}", "vmv.v.v {1}, v0",
         in(vreg) x, out(vreg) y, out("v0") m, options(nostack));
    let _ = m;
    y
}

// CHECK-LABEL: lateout_vreg:
// CHECK: #APP
// CHECK: vmv.v.v v{{[0-9]+}}, v{{[0-9]+}}
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn lateout_vreg(x: i8x16) -> i8x16 {
    let y: i8x16;
    asm!("vmv.v.v {0}, {1}", lateout(vreg) y, in(vreg) x, options(nostack));
    y
}

// CHECK-LABEL: inout_vreg:
// CHECK: #APP
// CHECK: vadd.vv v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn inout_vreg(x: i8x16, y: i8x16) -> i8x16 {
    let mut acc = x;
    asm!("vadd.vv {0}, {0}, {1}", inout(vreg) acc, in(vreg) y, options(nostack));
    acc
}
