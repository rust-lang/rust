//@ add-core-stubs
//@ revisions: powerpc powerpc_altivec powerpc_vsx powerpc64 powerpc64_vsx
//@ assembly-output: emit-asm
//@[powerpc] compile-flags: --target powerpc-unknown-linux-gnu
//@[powerpc] needs-llvm-components: powerpc
//@[powerpc_altivec] compile-flags: --target powerpc-unknown-linux-gnu -C target-feature=+altivec --cfg altivec
//@[powerpc_altivec] needs-llvm-components: powerpc
//@[powerpc_vsx] compile-flags: --target powerpc-unknown-linux-gnu -C target-feature=+altivec,+vsx --cfg altivec --cfg vsx
//@[powerpc_vsx] needs-llvm-components: powerpc
//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu --cfg altivec
//@[powerpc64] needs-llvm-components: powerpc
//@[powerpc64_vsx] compile-flags: --target powerpc64-unknown-linux-gnu -C target-feature=+vsx --cfg altivec --cfg vsx
//@[powerpc64_vsx] needs-llvm-components: powerpc
//@ compile-flags: -Zmerge-functions=disabled

#![feature(no_core, repr_simd, asm_experimental_arch)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

extern crate minicore;
use minicore::*;

#[cfg_attr(altivec, cfg(not(target_feature = "altivec")))]
#[cfg_attr(not(altivec), cfg(target_feature = "altivec"))]
compile_error!("altivec cfg and target feature mismatch");
#[cfg_attr(vsx, cfg(not(target_feature = "vsx")))]
#[cfg_attr(not(vsx), cfg(target_feature = "vsx"))]
compile_error!("vsx cfg and target feature mismatch");

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

macro_rules! check_reg { ($func:ident, $ty:ty, $rego:tt, $regc:tt, $mov:literal) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        let y;
        asm!(concat!($mov, " ", $rego, ", ", $rego), lateout($regc) y, in($regc) x);
        y
    }
};}

// CHECK-LABEL: reg_i8:
// CHECK: #APP
// CHECK: mr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i8, i8, reg, "mr");

// CHECK-LABEL: reg_i16:
// CHECK: #APP
// CHECK: mr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i16, i16, reg, "mr");

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// CHECK: mr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i32, i32, reg, "mr");

// powerpc64-LABEL: reg_i64:
// powerpc64: #APP
// powerpc64: mr {{[0-9]+}}, {{[0-9]+}}
// powerpc64: #NO_APP
#[cfg(powerpc64)]
check!(reg_i64, i64, reg, "mr");

// CHECK-LABEL: reg_i8_nz:
// CHECK: #APP
// CHECK: mr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i8_nz, i8, reg_nonzero, "mr");

// CHECK-LABEL: reg_i16_nz:
// CHECK: #APP
// CHECK: mr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i16_nz, i16, reg_nonzero, "mr");

// CHECK-LABEL: reg_i32_nz:
// CHECK: #APP
// CHECK: mr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i32_nz, i32, reg_nonzero, "mr");

// powerpc64-LABEL: reg_i64_nz:
// powerpc64: #APP
// powerpc64: mr {{[0-9]+}}, {{[0-9]+}}
// powerpc64: #NO_APP
#[cfg(powerpc64)]
check!(reg_i64_nz, i64, reg_nonzero, "mr");

// CHECK-LABEL: reg_f32:
// CHECK: #APP
// CHECK: fmr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f32, f32, freg, "fmr");

// CHECK-LABEL: reg_f64:
// CHECK: #APP
// CHECK: fmr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f64, f64, freg, "fmr");

// powerpc_altivec-LABEL: vreg_i8x16:
// powerpc_altivec: #APP
// powerpc_altivec: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc_altivec: #NO_APP
// powerpc64-LABEL: vreg_i8x16:
// powerpc64: #APP
// powerpc64: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc64: #NO_APP
#[cfg(altivec)]
check!(vreg_i8x16, i8x16, vreg, "vmr");

// powerpc_altivec-LABEL: vreg_i16x8:
// powerpc_altivec: #APP
// powerpc_altivec: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc_altivec: #NO_APP
// powerpc64-LABEL: vreg_i16x8:
// powerpc64: #APP
// powerpc64: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc64: #NO_APP
#[cfg(altivec)]
check!(vreg_i16x8, i16x8, vreg, "vmr");

// powerpc_altivec-LABEL: vreg_i32x4:
// powerpc_altivec: #APP
// powerpc_altivec: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc_altivec: #NO_APP
// powerpc64-LABEL: vreg_i32x4:
// powerpc64: #APP
// powerpc64: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc64: #NO_APP
#[cfg(altivec)]
check!(vreg_i32x4, i32x4, vreg, "vmr");

// powerpc_vsx-LABEL: vreg_i64x2:
// powerpc_vsx: #APP
// powerpc_vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc_vsx: #NO_APP
// powerpc64_vsx-LABEL: vreg_i64x2:
// powerpc64_vsx: #APP
// powerpc64_vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc64_vsx: #NO_APP
#[cfg(vsx)]
check!(vreg_i64x2, i64x2, vreg, "vmr");

// powerpc_altivec-LABEL: vreg_f32x4:
// powerpc_altivec: #APP
// powerpc_altivec: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc_altivec: #NO_APP
// powerpc64-LABEL: vreg_f32x4:
// powerpc64: #APP
// powerpc64: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc64: #NO_APP
#[cfg(altivec)]
check!(vreg_f32x4, f32x4, vreg, "vmr");

// powerpc_vsx-LABEL: vreg_f64x2:
// powerpc_vsx: #APP
// powerpc_vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc_vsx: #NO_APP
// powerpc64_vsx-LABEL: vreg_f64x2:
// powerpc64_vsx: #APP
// powerpc64_vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc64_vsx: #NO_APP
#[cfg(vsx)]
check!(vreg_f64x2, f64x2, vreg, "vmr");

// powerpc_vsx-LABEL: vreg_f32:
// powerpc_vsx: #APP
// powerpc_vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc_vsx: #NO_APP
// powerpc64_vsx-LABEL: vreg_f32:
// powerpc64_vsx: #APP
// powerpc64_vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc64_vsx: #NO_APP
#[cfg(vsx)]
check!(vreg_f32, f32, vreg, "vmr");

// powerpc_vsx-LABEL: vreg_f64:
// powerpc_vsx: #APP
// powerpc_vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc_vsx: #NO_APP
// powerpc64_vsx-LABEL: vreg_f64:
// powerpc64_vsx: #APP
// powerpc64_vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// powerpc64_vsx: #NO_APP
#[cfg(vsx)]
check!(vreg_f64, f64, vreg, "vmr");

// CHECK-LABEL: reg_i8_r0:
// CHECK: #APP
// CHECK: mr 0, 0
// CHECK: #NO_APP
check_reg!(reg_i8_r0, i8, "0", "0", "mr");

// CHECK-LABEL: reg_i16_r0:
// CHECK: #APP
// CHECK: mr 0, 0
// CHECK: #NO_APP
check_reg!(reg_i16_r0, i16, "0", "0", "mr");

// CHECK-LABEL: reg_i32_r0:
// CHECK: #APP
// CHECK: mr 0, 0
// CHECK: #NO_APP
check_reg!(reg_i32_r0, i32, "0", "0", "mr");

// powerpc64-LABEL: reg_i64_r0:
// powerpc64: #APP
// powerpc64: mr 0, 0
// powerpc64: #NO_APP
#[cfg(powerpc64)]
check_reg!(reg_i64_r0, i64, "0", "0", "mr");

// CHECK-LABEL: reg_i8_r18:
// CHECK: #APP
// CHECK: mr 18, 18
// CHECK: #NO_APP
check_reg!(reg_i8_r18, i8, "18", "18", "mr");

// CHECK-LABEL: reg_i16_r18:
// CHECK: #APP
// CHECK: mr 18, 18
// CHECK: #NO_APP
check_reg!(reg_i16_r18, i16, "18", "18", "mr");

// CHECK-LABEL: reg_i32_r18:
// CHECK: #APP
// CHECK: mr 18, 18
// CHECK: #NO_APP
check_reg!(reg_i32_r18, i32, "18", "18", "mr");

// powerpc64-LABEL: reg_i64_r18:
// powerpc64: #APP
// powerpc64: mr 18, 18
// powerpc64: #NO_APP
#[cfg(powerpc64)]
check_reg!(reg_i64_r18, i64, "18", "18", "mr");

// CHECK-LABEL: reg_f32_f0:
// CHECK: #APP
// CHECK: fmr 0, 0
// CHECK: #NO_APP
check_reg!(reg_f32_f0, f32, "0", "f0", "fmr");

// CHECK-LABEL: reg_f64_f0:
// CHECK: #APP
// CHECK: fmr 0, 0
// CHECK: #NO_APP
check_reg!(reg_f64_f0, f64, "0", "f0", "fmr");

// CHECK-LABEL: reg_f32_f18:
// CHECK: #APP
// CHECK: fmr 18, 18
// CHECK: #NO_APP
check_reg!(reg_f32_f18, f32, "18", "f18", "fmr");

// CHECK-LABEL: reg_f64_f18:
// CHECK: #APP
// CHECK: fmr 18, 18
// CHECK: #NO_APP
check_reg!(reg_f64_f18, f64, "18", "f18", "fmr");

// powerpc_altivec-LABEL: vreg_i8x16_v0:
// powerpc_altivec: #APP
// powerpc_altivec: vmr 0, 0
// powerpc_altivec: #NO_APP
// powerpc64-LABEL: vreg_i8x16_v0:
// powerpc64: #APP
// powerpc64: vmr 0, 0
// powerpc64: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_i8x16_v0, i8x16, "0", "v0", "vmr");

// powerpc_altivec-LABEL: vreg_i16x8_v0:
// powerpc_altivec: #APP
// powerpc_altivec: vmr 0, 0
// powerpc_altivec: #NO_APP
// powerpc64-LABEL: vreg_i16x8_v0:
// powerpc64: #APP
// powerpc64: vmr 0, 0
// powerpc64: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_i16x8_v0, i16x8, "0", "v0", "vmr");

// powerpc_altivec-LABEL: vreg_i32x4_v0:
// powerpc_altivec: #APP
// powerpc_altivec: vmr 0, 0
// powerpc_altivec: #NO_APP
// powerpc64-LABEL: vreg_i32x4_v0:
// powerpc64: #APP
// powerpc64: vmr 0, 0
// powerpc64: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_i32x4_v0, i32x4, "0", "v0", "vmr");

// powerpc_vsx-LABEL: vreg_i64x2_v0:
// powerpc_vsx: #APP
// powerpc_vsx: vmr 0, 0
// powerpc_vsx: #NO_APP
// powerpc64_vsx-LABEL: vreg_i64x2_v0:
// powerpc64_vsx: #APP
// powerpc64_vsx: vmr 0, 0
// powerpc64_vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_i64x2_v0, i64x2, "0", "v0", "vmr");

// powerpc_altivec-LABEL: vreg_f32x4_v0:
// powerpc_altivec: #APP
// powerpc_altivec: vmr 0, 0
// powerpc_altivec: #NO_APP
// powerpc64-LABEL: vreg_f32x4_v0:
// powerpc64: #APP
// powerpc64: vmr 0, 0
// powerpc64: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_f32x4_v0, f32x4, "0", "v0", "vmr");

// powerpc_vsx-LABEL: vreg_f64x2_v0:
// powerpc_vsx: #APP
// powerpc_vsx: vmr 0, 0
// powerpc_vsx: #NO_APP
// powerpc64_vsx-LABEL: vreg_f64x2_v0:
// powerpc64_vsx: #APP
// powerpc64_vsx: vmr 0, 0
// powerpc64_vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f64x2_v0, f64x2, "0", "v0", "vmr");

// powerpc_vsx-LABEL: vreg_f32_v0:
// powerpc_vsx: #APP
// powerpc_vsx: vmr 0, 0
// powerpc_vsx: #NO_APP
// powerpc64_vsx-LABEL: vreg_f32_v0:
// powerpc64_vsx: #APP
// powerpc64_vsx: vmr 0, 0
// powerpc64_vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f32_v0, f32, "0", "v0", "vmr");

// powerpc_vsx-LABEL: vreg_f64_v0:
// powerpc_vsx: #APP
// powerpc_vsx: vmr 0, 0
// powerpc_vsx: #NO_APP
// powerpc64_vsx-LABEL: vreg_f64_v0:
// powerpc64_vsx: #APP
// powerpc64_vsx: vmr 0, 0
// powerpc64_vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f64_v0, f64, "0", "v0", "vmr");

// powerpc_altivec-LABEL: vreg_i8x16_v18:
// powerpc_altivec: #APP
// powerpc_altivec: vmr 18, 18
// powerpc_altivec: #NO_APP
// powerpc64-LABEL: vreg_i8x16_v18:
// powerpc64: #APP
// powerpc64: vmr 18, 18
// powerpc64: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_i8x16_v18, i8x16, "18", "v18", "vmr");

// powerpc_altivec-LABEL: vreg_i16x8_v18:
// powerpc_altivec: #APP
// powerpc_altivec: vmr 18, 18
// powerpc_altivec: #NO_APP
// powerpc64-LABEL: vreg_i16x8_v18:
// powerpc64: #APP
// powerpc64: vmr 18, 18
// powerpc64: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_i16x8_v18, i16x8, "18", "v18", "vmr");

// powerpc_altivec-LABEL: vreg_i32x4_v18:
// powerpc_altivec: #APP
// powerpc_altivec: vmr 18, 18
// powerpc_altivec: #NO_APP
// powerpc64-LABEL: vreg_i32x4_v18:
// powerpc64: #APP
// powerpc64: vmr 18, 18
// powerpc64: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_i32x4_v18, i32x4, "18", "v18", "vmr");

// powerpc_vsx-LABEL: vreg_i64x2_v18:
// powerpc_vsx: #APP
// powerpc_vsx: vmr 18, 18
// powerpc_vsx: #NO_APP
// powerpc64_vsx-LABEL: vreg_i64x2_v18:
// powerpc64_vsx: #APP
// powerpc64_vsx: vmr 18, 18
// powerpc64_vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_i64x2_v18, i64x2, "18", "v18", "vmr");

// powerpc_altivec-LABEL: vreg_f32x4_v18:
// powerpc_altivec: #APP
// powerpc_altivec: vmr 18, 18
// powerpc_altivec: #NO_APP
// powerpc64-LABEL: vreg_f32x4_v18:
// powerpc64: #APP
// powerpc64: vmr 18, 18
// powerpc64: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_f32x4_v18, f32x4, "18", "v18", "vmr");

// powerpc_vsx-LABEL: vreg_f64x2_v18:
// powerpc_vsx: #APP
// powerpc_vsx: vmr 18, 18
// powerpc_vsx: #NO_APP
// powerpc64_vsx-LABEL: vreg_f64x2_v18:
// powerpc64_vsx: #APP
// powerpc64_vsx: vmr 18, 18
// powerpc64_vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f64x2_v18, f64x2, "18", "v18", "vmr");

// powerpc_vsx-LABEL: vreg_f32_v18:
// powerpc_vsx: #APP
// powerpc_vsx: vmr 18, 18
// powerpc_vsx: #NO_APP
// powerpc64_vsx-LABEL: vreg_f32_v18:
// powerpc64_vsx: #APP
// powerpc64_vsx: vmr 18, 18
// powerpc64_vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f32_v18, f32, "18", "v18", "vmr");

// powerpc_vsx-LABEL: vreg_f64_v18:
// powerpc_vsx: #APP
// powerpc_vsx: vmr 18, 18
// powerpc_vsx: #NO_APP
// powerpc64_vsx-LABEL: vreg_f64_v18:
// powerpc64_vsx: #APP
// powerpc64_vsx: vmr 18, 18
// powerpc64_vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f64_v18, f64, "18", "v18", "vmr");
