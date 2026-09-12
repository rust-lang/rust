// ignore-tidy-file-linelength (some revision //@ lines are over 100 chars long)

//@ add-minicore
//@ revisions: powerpc powerpc_altivec powerpc_vsx powerpc_power9 powerpc64 powerpc64_vsx powerpc64_power9 powerpc64le powerpc64le_power9
//@ assembly-output: emit-asm

//@[powerpc] compile-flags: --target powerpc-unknown-linux-gnu
//@[powerpc] needs-llvm-components: powerpc
//@[powerpc_altivec] compile-flags: --target powerpc-unknown-linux-gnu -C target-feature=+altivec --cfg altivec
//@[powerpc_altivec] needs-llvm-components: powerpc
//@[powerpc_altivec] filecheck-flags: --check-prefix altivec
//@[powerpc_vsx] compile-flags: --target powerpc-unknown-linux-gnu -C target-feature=+altivec,+vsx --cfg altivec --cfg vsx
//@[powerpc_vsx] needs-llvm-components: powerpc
//@[powerpc_vsx] filecheck-flags: --check-prefix altivec --check-prefix vsx
//@[powerpc_power9] compile-flags: --target powerpc-unknown-linux-gnu -C target-feature=+altivec,+vsx,+power9-vector --cfg altivec --cfg vsx --cfg power9 --cfg power9be
//@[powerpc_power9] needs-llvm-components: powerpc
//@[powerpc_power9] filecheck-flags: --check-prefix altivec --check-prefix vsx --check-prefix power9 --check-prefix power9be

//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu --cfg altivec
//@[powerpc64] needs-llvm-components: powerpc
//@[powerpc64] filecheck-flags: --check-prefix altivec
//@[powerpc64_vsx] compile-flags: --target powerpc64-unknown-linux-gnu -C target-feature=+vsx --cfg powerpc64 --cfg altivec --cfg vsx
//@[powerpc64_vsx] needs-llvm-components: powerpc
//@[powerpc64_vsx] filecheck-flags: --check-prefix powerpc64 --check-prefix altivec --check-prefix vsx
//@[powerpc64_power9] compile-flags: --target powerpc64-unknown-linux-gnu -C target-feature=+vsx,+power9-vector --cfg powerpc64 --cfg altivec --cfg vsx --cfg power9 --cfg power9be
//@[powerpc64_power9] needs-llvm-components: powerpc
//@[powerpc64_power9] filecheck-flags: --check-prefix powerpc64 --check-prefix altivec --check-prefix vsx --check-prefix power9 --check-prefix=power9be

//@[powerpc64le] compile-flags: --target powerpc64le-unknown-linux-gnu --cfg powerpc64 --cfg altivec --cfg vsx
//@[powerpc64le] needs-llvm-components: powerpc
//@[powerpc64le] filecheck-flags: --check-prefix powerpc64 --check-prefix altivec --check-prefix vsx
//@[powerpc64le_power9] compile-flags: --target powerpc64le-unknown-linux-gnu -C target-feature=+power9-vector --cfg powerpc64 --cfg altivec --cfg vsx --cfg power9
//@[powerpc64le_power9] needs-llvm-components: powerpc
//@[powerpc64le_power9] filecheck-flags: --check-prefix powerpc64 --check-prefix altivec --check-prefix vsx --check-prefix power9

//@ compile-flags: -Zmerge-functions=disabled -Copt-level=3
//@ compile-flags: --check-cfg=cfg(altivec,vsx,power9,power9be)
// PowerPC `f16` was broken before LLVM 22
//@ min-llvm-version: 22

#![feature(no_core, f16)]
#![cfg_attr(vsx, feature(f128))]
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
#[cfg_attr(power9, cfg(not(target_feature = "power9-vector")))]
#[cfg_attr(not(power9), cfg(target_feature = "power9-vector"))]
compile_error!("power9-vector cfg and target feature mismatch");
#[cfg_attr(power9be, cfg(not(all(target_feature = "power9-vector", target_endian = "big"))))]
#[cfg_attr(not(power9be), cfg(all(target_feature = "power9-vector", target_endian = "big")))]
compile_error!("power9be cfg and target feature mismatch");

// Check floating point scalars are put in the right vector lane. This uses power9 for consistent
// assembly between powerpc64le and big-endian powerpc/powerpc64.

// power9-LABEL: f32_to_f64:
// power9: .cfi_startproc
// power9-NEXT: xscvdpspn [[#INPUT:]], 1
// powerpc64le_power9-NEXT: vmrgow [[#INPUT - 32]], [[#INPUT - 32]], [[#INPUT - 32]]
// power9-NEXT: #APP
// power9-NEXT: xscvspdp 1, [[#INPUT]]
// power9-NEXT: #NO_APP
// power9-NEXT: blr
#[cfg(power9)]
#[unsafe(no_mangle)]
pub extern "C" fn f32_to_f64(x: f32) -> f64 {
    let res;
    unsafe {
        asm!("xscvspdp {}, {}", out(vsreg) res, in(vsreg) x, options(pure, nostack, nomem));
    };
    res
}

// power9-LABEL: f64_to_f32:
// power9: .cfi_startproc
// power9-NEXT: #APP
// power9-NEXT: xscvdpsp [[#OUTPUT:]], 1
// power9-NEXT: #NO_APP
// power9-NEXT: xscvspdpn 1, [[#OUTPUT]]
// power9-NEXT: blr
#[cfg(power9)]
#[unsafe(no_mangle)]
pub extern "C" fn f64_to_f32(x: f64) -> f32 {
    let res;
    unsafe {
        asm!("xscvdpsp {}, {}", out(vsreg) res, in(vsreg) x, options(pure, nostack, nomem));
    };
    res
}

// power9-LABEL: f64_to_f128:
// power9: .cfi_startproc
// power9-NEXT: xscpsgndp [[#INPUT:]], 1, 1
// power9-NEXT: #APP
// power9-NEXT: xscvdpqp 2, [[#INPUT - 32]]
// power9-NEXT: #NO_APP
// power9-NEXT: blr
#[cfg(power9)]
#[unsafe(no_mangle)]
pub extern "C" fn f64_to_f128(x: f64) -> f128 {
    let res;
    unsafe {
        asm!("xscvdpqp {}, {}", out(vreg) res, in(vreg) x, options(pure, nostack, nomem));
    };
    res
}

// FIXME(f16): This test will need to be updated if/when the `f16` ABI gets standardised.
// (see https://github.com/llvm/llvm-project/pull/196559)
// power9-LABEL: f64_to_f16:
// power9: .cfi_startproc
// powerpc_power9-NEXT: stwu 1, -[[#STACK:]](1)
// powerpc_power9-NEXT: .cfi_def_cfa_offset [[#STACK]]
// powerpc64le_power9-NEXT: li [[#INDEX:]], 8
// power9-NEXT: #APP
// power9-NEXT: xscvdphp [[#OUTPUT:]], 1
// power9-NEXT: #NO_APP
// power9be-NEXT: stxv [[#OUTPUT]], [[#%d,OFFSET:]](1)
// power9be-NEXT: lhz 3, [[#OFFSET + 6]](1)
// powerpc_power9-NEXT: addi 1, 1, [[#STACK]]
// powerpc64le_power9-NEXT: vextuhrx 3, [[#INDEX]], [[#OUTPUT - 32]]
// power9-NEXT: blr
#[cfg(power9)]
#[unsafe(no_mangle)]
pub extern "C" fn f64_to_f16(x: f64) -> f16 {
    let res;
    unsafe {
        asm!("xscvdphp {}, {}", out(vsreg) res, in(vsreg) x, options(pure, nostack, nomem));
    };
    res
}

macro_rules! check { ($func:ident, $ty:ty, $class:ident, $mov:literal) => {
    #[unsafe(no_mangle)]
    // FIXME(f128): Replace `&$ty` with `$ty` once
    // https://github.com/llvm/llvm-project/issues/213355 is fixed.
    pub unsafe fn $func(x: &$ty) -> $ty {
        let y;
        asm!(concat!($mov," {}, {}"), out($class) y, in($class) *x);
        y
    }
};}

macro_rules! check_reg { ($func:ident, $ty:ty, $rego:tt, $regc:tt, $mov:literal) => {
    #[unsafe(no_mangle)]
    // FIXME(f128): Replace `&$ty` with `$ty` once
    // https://github.com/llvm/llvm-project/issues/213355 is fixed.
    pub unsafe fn $func(x: &$ty) -> $ty {
        let y;
        asm!(concat!($mov, " ", $rego, ", ", $rego), lateout($regc) y, in($regc) *x);
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

// vsx-LABEL: vreg_f16x8:
// vsx: #APP
// vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vreg_f16x8, f16x8, vreg, "vmr");

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

// vsx-LABEL: vreg_f16:
// vsx: #APP
// vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vreg_f16, f16, vreg, "vmr");

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

// vsx-LABEL: vreg_f128:
// vsx: #APP
// vsx: vmr {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vreg_f128, f128, vreg, "vmr");

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

// vsx-LABEL: vsreg_f16x8:
// vsx: #APP
// vsx: xvsqrtdp {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vsreg_f16x8, f16x8, vsreg, "xvsqrtdp");

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

// vsx-LABEL: vsreg_f16:
// vsx: #APP
// vsx: xvsqrtdp {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vsreg_f16, f16, vsreg, "xvsqrtdp");

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

// vsx-LABEL: vsreg_f128:
// vsx: #APP
// vsx: xvsqrtdp {{[0-9]+}}, {{[0-9]+}}
// vsx: #NO_APP
#[cfg(vsx)]
check!(vsreg_f128, f128, vsreg, "xvsqrtdp");

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

// vsx-LABEL: vreg_f16x8_v0:
// vsx: #APP
// vsx: vmr 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f16x8_v0, f16x8, "0", "v0", "vmr");

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

// vsx-LABEL: vreg_f16_v0:
// vsx: #APP
// vsx: vmr 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f16_v0, f16, "0", "v0", "vmr");

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

// vsx-LABEL: vreg_f128_v0:
// vsx: #APP
// vsx: vmr 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f128_v0, f128, "0", "v0", "vmr");

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

// vsx-LABEL: vreg_f16x8_v18:
// vsx: #APP
// vsx: vmr 18, 18
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f16x8_v18, f16x8, "18", "v18", "vmr");

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

// vsx-LABEL: vreg_f16_v18:
// vsx: #APP
// vsx: vmr 18, 18
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f16_v18, f16, "18", "v18", "vmr");

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

// vsx-LABEL: vreg_f128_v18:
// vsx: #APP
// vsx: vmr 18, 18
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vreg_f128_v18, f128, "18", "v18", "vmr");

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

// vsx-LABEL: vsreg_f16x8_vs0:
// vsx: #APP
// vsx: xvsqrtdp 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f16x8_vs0, f16x8, "0", "vs0", "xvsqrtdp");

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

// vsx-LABEL: vsreg_f16_vs0:
// vsx: #APP
// vsx: xvsqrtdp 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f16_vs0, f16, "0", "vs0", "xvsqrtdp");

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

// vsx-LABEL: vsreg_f128_vs0:
// vsx: #APP
// vsx: xvsqrtdp 0, 0
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f128_vs0, f128, "0", "vs0", "xvsqrtdp");

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

// vsx-LABEL: vsreg_f16x8_v40:
// vsx: #APP
// vsx: xvsqrtdp 40, 40
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f16x8_v40, f16x8, "40", "vs40", "xvsqrtdp");

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

// vsx-LABEL: vsreg_f16_v40:
// vsx: #APP
// vsx: xvsqrtdp 40, 40
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f16_v40, f16, "40", "vs40", "xvsqrtdp");

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

// vsx-LABEL: vsreg_f128_v40:
// vsx: #APP
// vsx: xvsqrtdp 40, 40
// vsx: #NO_APP
#[cfg(vsx)]
check_reg!(vsreg_f128_v40, f128, "40", "vs40", "xvsqrtdp");
