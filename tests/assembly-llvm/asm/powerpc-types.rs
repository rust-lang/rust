// ignore-tidy-file-linelength (some revision //@ lines are over 100 chars long)

//@ add-minicore
//@ revisions: powerpc powerpc_altivec powerpc_vsx powerpc_power8 powerpc64 powerpc64_vsx powerpc64_power8 powerpc64le
//@ assembly-output: emit-asm
//@[powerpc] compile-flags: --target powerpc-unknown-linux-gnu
//@[powerpc] needs-llvm-components: powerpc
//@[powerpc_altivec] compile-flags: --target powerpc-unknown-linux-gnu -C target-feature=+altivec --cfg altivec
//@[powerpc_altivec] needs-llvm-components: powerpc
//@[powerpc_altivec] filecheck-flags: --check-prefix altivec
//@[powerpc_vsx] compile-flags: --target powerpc-unknown-linux-gnu -C target-feature=+altivec,+vsx --cfg altivec --cfg vsx
//@[powerpc_vsx] needs-llvm-components: powerpc
//@[powerpc_vsx] filecheck-flags: --check-prefix altivec --check-prefix vsx
//@[powerpc_power8] compile-flags: --target powerpc-unknown-linux-gnu -C target-feature=+altivec,+vsx,+power8-vector --cfg altivec --cfg vsx --cfg power8
//@[powerpc_power8] needs-llvm-components: powerpc
//@[powerpc_power8] filecheck-flags: --check-prefix altivec --check-prefix vsx --check-prefix power8
//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu --cfg altivec
//@[powerpc64] needs-llvm-components: powerpc
//@[powerpc64] filecheck-flags: --check-prefix altivec
//@[powerpc64_vsx] compile-flags: --target powerpc64-unknown-linux-gnu -C target-feature=+vsx --cfg powerpc64 --cfg altivec --cfg vsx
//@[powerpc64_vsx] needs-llvm-components: powerpc
//@[powerpc64_vsx] filecheck-flags: --check-prefix powerpc64 --check-prefix altivec --check-prefix vsx
//@[powerpc64_power8] compile-flags: --target powerpc64-unknown-linux-gnu -C target-feature=+vsx,+power8-vector --cfg powerpc64 --cfg altivec --cfg vsx --cfg power8
//@[powerpc64_power8] needs-llvm-components: powerpc
//@[powerpc64_power8] filecheck-flags: --check-prefix powerpc64 --check-prefix altivec --check-prefix vsx --check-prefix power8
//@[powerpc64le] compile-flags: --target powerpc64le-unknown-linux-gnu --cfg powerpc64 --cfg altivec --cfg vsx --cfg power8
//@[powerpc64le] needs-llvm-components: powerpc
//@[powerpc64le] filecheck-flags: --check-prefix powerpc64 --check-prefix altivec --check-prefix vsx --check-prefix power8
//@ compile-flags: -Zmerge-functions=disabled -O
//@ compile-flags: --check-cfg=cfg(altivec,vsx,power8)

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types, unused_imports)]
#![deny(unexpected_cfgs)]

extern crate minicore;
use minicore::simd::*;
use minicore::*;

#[cfg_attr(powerpc64, cfg(not(target_arch = "powerpc64")))]
#[cfg_attr(not(powerpc64), cfg(target_arch = "powerpc64"))]
compile_error!("powerpc64 cfg and target arch mismatch");
#[cfg_attr(altivec, cfg(not(target_feature = "altivec")))]
#[cfg_attr(not(altivec), cfg(target_feature = "altivec"))]
compile_error!("altivec cfg and target feature mismatch");
#[cfg_attr(vsx, cfg(not(target_feature = "vsx")))]
#[cfg_attr(not(vsx), cfg(target_feature = "vsx"))]
compile_error!("vsx cfg and target feature mismatch");
#[cfg_attr(power8, cfg(not(target_feature = "power8-vector")))]
#[cfg_attr(not(power8), cfg(target_feature = "power8-vector"))]
compile_error!("power8-vector cfg and target feature mismatch");

// Check floating point scalars are put in the right vector lane. This uses power8 for consistent
// assembly between powerpc64le and big-endian powerpc/powerpc64.

// power8-LABEL: f32_to_f64:
// power8: .cfi_startproc
// power8-NEXT: xscvdpspn [[#INPUT:]], 1
// powerpc64le-NEXT: vmrgow [[#INPUT - 32]], [[#INPUT - 32]], [[#INPUT - 32]]
// power8-NEXT: #APP
// power8-NEXT: xscvspdp 1, [[#INPUT]]
// power8-NEXT: #NO_APP
// power8-NEXT: blr
#[cfg(power8)]
#[unsafe(no_mangle)]
pub fn f32_to_f64(x: f32) -> f64 {
    let res;
    unsafe {
        asm!("xscvspdp {}, {}", out(vsreg) res, in(vsreg) x, options(pure, nostack, nomem));
    };
    res
}

// power8-LABEL: f64_to_f32:
// power8: .cfi_startproc
// power8-NEXT: #APP
// power8-NEXT: xscvdpsp [[#OUTPUT:]], 1
// power8-NEXT: #NO_APP
// power8-NEXT: xscvspdpn 1, [[#OUTPUT]]
// power8-NEXT: blr
#[cfg(power8)]
#[unsafe(no_mangle)]
pub fn f64_to_f32(x: f64) -> f32 {
    let res;
    unsafe {
        asm!("xscvdpsp {}, {}", out(vsreg) res, in(vsreg) x, options(pure, nostack, nomem));
    };
    res
}

macro_rules! check { ($func:ident, $ty:ty, $class:ident, $mov:literal) => {
    #[unsafe(no_mangle)]
    pub unsafe fn $func(x: $ty) -> $ty {
        let y;
        asm!(concat!($mov," {}, {}"), out($class) y, in($class) x);
        y
    }
};}

macro_rules! check_reg { ($func:ident, $ty:ty, $rego:tt, $regc:tt, $mov:literal) => {
    #[unsafe(no_mangle)]
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

// altivec-LABEL: vreg_i8x16:
// altivec: #APP
// altivec: vmr {{[0-9]+}}, {{[0-9]+}}
// altivec: #NO_APP
#[cfg(altivec)]
check!(vreg_i8x16, i8x16, vreg, "vmr");

// altivec-LABEL: vreg_i16x8:
// altivec: #APP
// altivec: vmr {{[0-9]+}}, {{[0-9]+}}
// altivec: #NO_APP
#[cfg(altivec)]
check!(vreg_i16x8, i16x8, vreg, "vmr");

// altivec-LABEL: vreg_i32x4:
// altivec: #APP
// altivec: vmr {{[0-9]+}}, {{[0-9]+}}
// altivec: #NO_APP
#[cfg(altivec)]
check!(vreg_i32x4, i32x4, vreg, "vmr");

// vsx-LABEL: vreg_i64x2:
// vsx: #APP
// vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vreg_i64x2, i64x2, vreg, "vmr");

// altivec-LABEL: vreg_f32x4:
// altivec: #APP
// altivec: vmr {{[0-9]+}}, {{[0-9]+}}
// altivec: #NO_APP
#[cfg(altivec)]
check!(vreg_f32x4, f32x4, vreg, "vmr");

// vsx-LABEL: vreg_f64x2:
// vsx: #APP
// vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vreg_f64x2, f64x2, vreg, "vmr");

// vsx-LABEL: vreg_f32:
// vsx: #APP
// vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vreg_f32, f32, vreg, "vmr");

// vsx-LABEL: vreg_f64:
// vsx: #APP
// vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vreg_f64, f64, vreg, "vmr");

// vsx-LABEL: vsreg_i8x16:
// vsx: #APP
// vsx: xvsqrtdp {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vsreg_i8x16, i8x16, vsreg, "xvsqrtdp");

// vsx-LABEL: vsreg_i16x8:
// vsx: #APP
// vsx: xvsqrtdp {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vsreg_i16x8, i16x8, vsreg, "xvsqrtdp");

// vsx-LABEL: vsreg_i32x4:
// vsx: #APP
// vsx: xvsqrtdp {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vsreg_i32x4, i32x4, vsreg, "xvsqrtdp");

// vsx-LABEL: vsreg_i64x2:
// vsx: #APP
// vsx: xvsqrtdp {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vsreg_i64x2, i64x2, vsreg, "xvsqrtdp");

// vsx-LABEL: vsreg_f32x4:
// vsx: #APP
// vsx: xvsqrtdp {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vsreg_f32x4, f32x4, vsreg, "xvsqrtdp");

// vsx-LABEL: vsreg_f64x2:
// vsx: #APP
// vsx: xvsqrtdp {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vsreg_f64x2, f64x2, vsreg, "xvsqrtdp");

// vsx-LABEL: vsreg_f32:
// vsx: #APP
// vsx: xvsqrtdp {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vsreg_f32, f32, vsreg, "xvsqrtdp");

// vsx-LABEL: vsreg_f64:
// vsx: #APP
// vsx: xvsqrtdp {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vsreg_f64, f64, vsreg, "xvsqrtdp");

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

// altivec-LABEL: vreg_i8x16_v0:
// altivec: #APP
// altivec: vmr 0, 0
// altivec: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_i8x16_v0, i8x16, "0", "v0", "vmr");

// altivec-LABEL: vreg_i16x8_v0:
// altivec: #APP
// altivec: vmr 0, 0
// altivec: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_i16x8_v0, i16x8, "0", "v0", "vmr");

// altivec-LABEL: vreg_i32x4_v0:
// altivec: #APP
// altivec: vmr 0, 0
// altivec: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_i32x4_v0, i32x4, "0", "v0", "vmr");

// vsx-LABEL: vreg_i64x2_v0:
// vsx: #APP
// vsx: vmr 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_i64x2_v0, i64x2, "0", "v0", "vmr");

// altivec-LABEL: vreg_f32x4_v0:
// altivec: #APP
// altivec: vmr 0, 0
// altivec: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_f32x4_v0, f32x4, "0", "v0", "vmr");

// vsx-LABEL: vreg_f64x2_v0:
// vsx: #APP
// vsx: vmr 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f64x2_v0, f64x2, "0", "v0", "vmr");

// vsx-LABEL: vreg_f32_v0:
// vsx: #APP
// vsx: vmr 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f32_v0, f32, "0", "v0", "vmr");

// vsx-LABEL: vreg_f64_v0:
// vsx: #APP
// vsx: vmr 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f64_v0, f64, "0", "v0", "vmr");

// altivec-LABEL: vreg_i8x16_v18:
// altivec: #APP
// altivec: vmr 18, 18
// altivec: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_i8x16_v18, i8x16, "18", "v18", "vmr");

// altivec-LABEL: vreg_i16x8_v18:
// altivec: #APP
// altivec: vmr 18, 18
// altivec: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_i16x8_v18, i16x8, "18", "v18", "vmr");

// altivec-LABEL: vreg_i32x4_v18:
// altivec: #APP
// altivec: vmr 18, 18
// altivec: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_i32x4_v18, i32x4, "18", "v18", "vmr");

// vsx-LABEL: vreg_i64x2_v18:
// vsx: #APP
// vsx: vmr 18, 18
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_i64x2_v18, i64x2, "18", "v18", "vmr");

// altivec-LABEL: vreg_f32x4_v18:
// altivec: #APP
// altivec: vmr 18, 18
// altivec: #NO_APP
#[cfg(altivec)]
check_reg!(vreg_f32x4_v18, f32x4, "18", "v18", "vmr");

// vsx-LABEL: vreg_f64x2_v18:
// vsx: #APP
// vsx: vmr 18, 18
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f64x2_v18, f64x2, "18", "v18", "vmr");

// vsx-LABEL: vreg_f32_v18:
// vsx: #APP
// vsx: vmr 18, 18
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f32_v18, f32, "18", "v18", "vmr");

// vsx-LABEL: vreg_f64_v18:
// vsx: #APP
// vsx: vmr 18, 18
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f64_v18, f64, "18", "v18", "vmr");

// vsx-LABEL: vsreg_i8x16_vs0:
// vsx: #APP
// vsx: xvsqrtdp 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_i8x16_vs0, i8x16, "0", "vs0", "xvsqrtdp");

// vsx-LABEL: vsreg_i16x8_vs0:
// vsx: #APP
// vsx: xvsqrtdp 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_i16x8_vs0, i16x8, "0", "vs0", "xvsqrtdp");

// vsx-LABEL: vsreg_i32x4_vs0:
// vsx: #APP
// vsx: xvsqrtdp 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_i32x4_vs0, i32x4, "0", "vs0", "xvsqrtdp");

// vsx-LABEL: vsreg_i64x2_vs0:
// vsx: #APP
// vsx: xvsqrtdp 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_i64x2_vs0, i64x2, "0", "vs0", "xvsqrtdp");

// vsx-LABEL: vsreg_f32x4_vs0:
// vsx: #APP
// vsx: xvsqrtdp 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f32x4_vs0, f32x4, "0", "vs0", "xvsqrtdp");

// vsx-LABEL: vsreg_f64x2_vs0:
// vsx: #APP
// vsx: xvsqrtdp 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f64x2_vs0, f64x2, "0", "vs0", "xvsqrtdp");

// vsx-LABEL: vsreg_f32_vs0:
// vsx: #APP
// vsx: xvsqrtdp 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f32_vs0, f32, "0", "vs0", "xvsqrtdp");

// vsx-LABEL: vsreg_f64_vs0:
// vsx: #APP
// vsx: xvsqrtdp 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f64_vs0, f64, "0", "vs0", "xvsqrtdp");

// vsx-LABEL: vsreg_i8x16_v40:
// vsx: #APP
// vsx: xvsqrtdp 40, 40
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_i8x16_v40, i8x16, "40", "vs40", "xvsqrtdp");

// vsx-LABEL: vsreg_i16x8_v40:
// vsx: #APP
// vsx: xvsqrtdp 40, 40
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_i16x8_v40, i16x8, "40", "vs40", "xvsqrtdp");

// vsx-LABEL: vsreg_i32x4_v40:
// vsx: #APP
// vsx: xvsqrtdp 40, 40
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_i32x4_v40, i32x4, "40", "vs40", "xvsqrtdp");

// vsx-LABEL: vsreg_i64x2_v40:
// vsx: #APP
// vsx: xvsqrtdp 40, 40
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_i64x2_v40, i64x2, "40", "vs40", "xvsqrtdp");

// vsx-LABEL: vsreg_f32x4_v40:
// vsx: #APP
// vsx: xvsqrtdp 40, 40
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f32x4_v40, f32x4, "40", "vs40", "xvsqrtdp");

// vsx-LABEL: vsreg_f64x2_v40:
// vsx: #APP
// vsx: xvsqrtdp 40, 40
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f64x2_v40, f64x2, "40", "vs40", "xvsqrtdp");

// vsx-LABEL: vsreg_f32_v40:
// vsx: #APP
// vsx: xvsqrtdp 40, 40
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f32_v40, f32, "40", "vs40", "xvsqrtdp");

// vsx-LABEL: vsreg_f64_v40:
// vsx: #APP
// vsx: xvsqrtdp 40, 40
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f64_v40, f64, "40", "vs40", "xvsqrtdp");
