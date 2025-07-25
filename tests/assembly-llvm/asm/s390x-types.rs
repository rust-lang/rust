//@ add-core-stubs
//@ revisions: s390x s390x_vector
//@ assembly-output: emit-asm
//@[s390x] compile-flags: --target s390x-unknown-linux-gnu
//@[s390x] needs-llvm-components: systemz
//@[s390x_vector] compile-flags: --target s390x-unknown-linux-gnu -C target-feature=+vector
//@[s390x_vector] needs-llvm-components: systemz
//@ compile-flags: -Zmerge-functions=disabled

#![feature(no_core, repr_simd, f128)]
#![cfg_attr(s390x_vector, feature(asm_experimental_reg))]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

extern crate minicore;
use minicore::*;

type ptr = *const i32;

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

impl Copy for i8x16 {}
impl Copy for i16x8 {}
impl Copy for i32x4 {}
impl Copy for i64x2 {}
impl Copy for f32x4 {}
impl Copy for f64x2 {}

extern "C" {
    fn extern_func();
    static extern_static: u8;
}

macro_rules! check { ($func:ident, $ty:ty, $class:ident, $mov:literal) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        let y;
        asm!(concat!($mov," {}, {}"), out($class) y, in($class) x);
        y
    }
};}

macro_rules! check_reg { ($func:ident, $ty:ty, $reg:tt, $mov:literal) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        let y;
        asm!(concat!($mov, " %", $reg, ", %", $reg), lateout($reg) y, in($reg) x);
        y
    }
};}

// CHECK-LABEL: sym_fn_32:
// CHECK: #APP
// CHECK: brasl %r14, extern_func
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_fn_32() {
    asm!("brasl %r14, {}", sym extern_func);
}

// CHECK-LABEL: sym_static:
// CHECK: #APP
// CHECK: brasl %r14, extern_static
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_static() {
    asm!("brasl %r14, {}", sym extern_static);
}

// CHECK-LABEL: reg_i8:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i8, i8, reg, "lgr");

// CHECK-LABEL: reg_i16:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i16, i16, reg, "lgr");

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i32, i32, reg, "lgr");

// CHECK-LABEL: reg_i64:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i64, i64, reg, "lgr");

// CHECK-LABEL: reg_i8_addr:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i8_addr, i8, reg_addr, "lgr");

// CHECK-LABEL: reg_i16_addr:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i16_addr, i16, reg_addr, "lgr");

// CHECK-LABEL: reg_i32_addr:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i32_addr, i32, reg_addr, "lgr");

// CHECK-LABEL: reg_i64_addr:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i64_addr, i64, reg_addr, "lgr");

// CHECK-LABEL: reg_f32:
// CHECK: #APP
// CHECK: ler %f{{[0-9]+}}, %f{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f32, f32, freg, "ler");

// CHECK-LABEL: reg_f64:
// CHECK: #APP
// CHECK: ldr %f{{[0-9]+}}, %f{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f64, f64, freg, "ldr");

// CHECK-LABEL: reg_ptr:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_ptr, ptr, reg, "lgr");

// s390x_vector-LABEL: vreg_i8x16:
// s390x_vector: #APP
// s390x_vector: vlr %v{{[0-9]+}}, %v{{[0-9]+}}
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check!(vreg_i8x16, i8x16, vreg, "vlr");

// s390x_vector-LABEL: vreg_i16x8:
// s390x_vector: #APP
// s390x_vector: vlr %v{{[0-9]+}}, %v{{[0-9]+}}
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check!(vreg_i16x8, i16x8, vreg, "vlr");

// s390x_vector-LABEL: vreg_i32x4:
// s390x_vector: #APP
// s390x_vector: vlr %v{{[0-9]+}}, %v{{[0-9]+}}
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check!(vreg_i32x4, i32x4, vreg, "vlr");

// s390x_vector-LABEL: vreg_i64x2:
// s390x_vector: #APP
// s390x_vector: vlr %v{{[0-9]+}}, %v{{[0-9]+}}
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check!(vreg_i64x2, i64x2, vreg, "vlr");

// s390x_vector-LABEL: vreg_f32x4:
// s390x_vector: #APP
// s390x_vector: vlr %v{{[0-9]+}}, %v{{[0-9]+}}
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check!(vreg_f32x4, f32x4, vreg, "vlr");

// s390x_vector-LABEL: vreg_f64x2:
// s390x_vector: #APP
// s390x_vector: vlr %v{{[0-9]+}}, %v{{[0-9]+}}
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check!(vreg_f64x2, f64x2, vreg, "vlr");

// s390x_vector-LABEL: vreg_i32:
// s390x_vector: #APP
// s390x_vector: vlr %v{{[0-9]+}}, %v{{[0-9]+}}
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check!(vreg_i32, i32, vreg, "vlr");

// s390x_vector-LABEL: vreg_i64:
// s390x_vector: #APP
// s390x_vector: vlr %v{{[0-9]+}}, %v{{[0-9]+}}
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check!(vreg_i64, i64, vreg, "vlr");

// s390x_vector-LABEL: vreg_i128:
// s390x_vector: #APP
// s390x_vector: vlr %v{{[0-9]+}}, %v{{[0-9]+}}
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check!(vreg_i128, i128, vreg, "vlr");

// s390x_vector-LABEL: vreg_f32:
// s390x_vector: #APP
// s390x_vector: vlr %v{{[0-9]+}}, %v{{[0-9]+}}
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check!(vreg_f32, f32, vreg, "vlr");

// s390x_vector-LABEL: vreg_f64:
// s390x_vector: #APP
// s390x_vector: vlr %v{{[0-9]+}}, %v{{[0-9]+}}
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check!(vreg_f64, f64, vreg, "vlr");

// s390x_vector-LABEL: vreg_f128:
// s390x_vector: #APP
// s390x_vector: vlr %v{{[0-9]+}}, %v{{[0-9]+}}
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check!(vreg_f128, f128, vreg, "vlr");

// CHECK-LABEL: r0_i8:
// CHECK: #APP
// CHECK: lr %r0, %r0
// CHECK: #NO_APP
check_reg!(r0_i8, i8, "r0", "lr");

// CHECK-LABEL: r0_i16:
// CHECK: #APP
// CHECK: lr %r0, %r0
// CHECK: #NO_APP
check_reg!(r0_i16, i16, "r0", "lr");

// CHECK-LABEL: r0_i32:
// CHECK: #APP
// CHECK: lr %r0, %r0
// CHECK: #NO_APP
check_reg!(r0_i32, i32, "r0", "lr");

// CHECK-LABEL: r0_i64:
// CHECK: #APP
// CHECK: lr %r0, %r0
// CHECK: #NO_APP
check_reg!(r0_i64, i64, "r0", "lr");

// CHECK-LABEL: f0_f32:
// CHECK: #APP
// CHECK: ler %f0, %f0
// CHECK: #NO_APP
check_reg!(f0_f32, f32, "f0", "ler");

// CHECK-LABEL: f0_f64:
// CHECK: #APP
// CHECK: ldr %f0, %f0
// CHECK: #NO_APP
check_reg!(f0_f64, f64, "f0", "ldr");

// s390x_vector-LABEL: v0_i8x16:
// s390x_vector: #APP
// s390x_vector: vlr %v0, %v0
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check_reg!(v0_i8x16, i8x16, "v0", "vlr");

// s390x_vector-LABEL: v0_i16x8:
// s390x_vector: #APP
// s390x_vector: vlr %v0, %v0
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check_reg!(v0_i16x8, i16x8, "v0", "vlr");

// s390x_vector-LABEL: v0_i32x4:
// s390x_vector: #APP
// s390x_vector: vlr %v0, %v0
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check_reg!(v0_i32x4, i32x4, "v0", "vlr");

// s390x_vector-LABEL: v0_i64x2:
// s390x_vector: #APP
// s390x_vector: vlr %v0, %v0
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check_reg!(v0_i64x2, i64x2, "v0", "vlr");

// s390x_vector-LABEL: v0_f32x4:
// s390x_vector: #APP
// s390x_vector: vlr %v0, %v0
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check_reg!(v0_f32x4, f32x4, "v0", "vlr");

// s390x_vector-LABEL: v0_f64x2:
// s390x_vector: #APP
// s390x_vector: vlr %v0, %v0
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check_reg!(v0_f64x2, f64x2, "v0", "vlr");

// s390x_vector-LABEL: v0_i32:
// s390x_vector: #APP
// s390x_vector: vlr %v0, %v0
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check_reg!(v0_i32, i32, "v0", "vlr");

// s390x_vector-LABEL: v0_i64:
// s390x_vector: #APP
// s390x_vector: vlr %v0, %v0
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check_reg!(v0_i64, i64, "v0", "vlr");

// s390x_vector-LABEL: v0_i128:
// s390x_vector: #APP
// s390x_vector: vlr %v0, %v0
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check_reg!(v0_i128, i128, "v0", "vlr");

// s390x_vector-LABEL: v0_f32:
// s390x_vector: #APP
// s390x_vector: vlr %v0, %v0
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check_reg!(v0_f32, f32, "v0", "vlr");

// s390x_vector-LABEL: v0_f64:
// s390x_vector: #APP
// s390x_vector: vlr %v0, %v0
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check_reg!(v0_f64, f64, "v0", "vlr");

// s390x_vector-LABEL: v0_f128:
// s390x_vector: #APP
// s390x_vector: vlr %v0, %v0
// s390x_vector: #NO_APP
#[cfg(s390x_vector)]
check_reg!(v0_f128, f128, "v0", "vlr");
