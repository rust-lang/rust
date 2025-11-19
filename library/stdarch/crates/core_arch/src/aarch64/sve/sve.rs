// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen2/spec/` and run the following command to re-generate this file:
//
// ```
// cargo run --bin=stdarch-gen2 -- crates/stdarch-gen2/spec
// ```
#![allow(improper_ctypes)]

#[cfg(test)]
use stdarch_test::assert_instr;

use super::*;

#[doc = "Absolute compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svacge[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facge))]
pub fn svacge_f32(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.facge.nxv4f32")]
        fn _svacge_f32(pg: svbool4_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svacge_f32(simd_cast(pg), op1, op2)) }
}
#[doc = "Absolute compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svacge[_n_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facge))]
pub fn svacge_n_f32(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svbool_t {
    svacge_f32(pg, op1, svdup_n_f32(op2))
}
#[doc = "Absolute compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svacge[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facge))]
pub fn svacge_f64(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.facge.nxv2f64")]
        fn _svacge_f64(pg: svbool2_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool2_t;
    }
    unsafe { simd_cast(_svacge_f64(simd_cast(pg), op1, op2)) }
}
#[doc = "Absolute compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svacge[_n_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facge))]
pub fn svacge_n_f64(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svbool_t {
    svacge_f64(pg, op1, svdup_n_f64(op2))
}
#[doc = "Absolute compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svacgt[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facgt))]
pub fn svacgt_f32(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.facgt.nxv4f32")]
        fn _svacgt_f32(pg: svbool4_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svacgt_f32(simd_cast(pg), op1, op2)) }
}
#[doc = "Absolute compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svacgt[_n_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facgt))]
pub fn svacgt_n_f32(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svbool_t {
    svacgt_f32(pg, op1, svdup_n_f32(op2))
}
#[doc = "Absolute compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svacgt[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facgt))]
pub fn svacgt_f64(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.facgt.nxv2f64")]
        fn _svacgt_f64(pg: svbool2_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool2_t;
    }
    unsafe { simd_cast(_svacgt_f64(simd_cast(pg), op1, op2)) }
}
#[doc = "Absolute compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svacgt[_n_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facgt))]
pub fn svacgt_n_f64(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svbool_t {
    svacgt_f64(pg, op1, svdup_n_f64(op2))
}
#[doc = "Absolute compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svacle[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facge))]
pub fn svacle_f32(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool_t {
    svacge_f32(pg, op2, op1)
}
#[doc = "Absolute compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svacle[_n_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facge))]
pub fn svacle_n_f32(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svbool_t {
    svacle_f32(pg, op1, svdup_n_f32(op2))
}
#[doc = "Absolute compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svacle[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facge))]
pub fn svacle_f64(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool_t {
    svacge_f64(pg, op2, op1)
}
#[doc = "Absolute compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svacle[_n_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facge))]
pub fn svacle_n_f64(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svbool_t {
    svacle_f64(pg, op1, svdup_n_f64(op2))
}
#[doc = "Absolute compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaclt[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facgt))]
pub fn svaclt_f32(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool_t {
    svacgt_f32(pg, op2, op1)
}
#[doc = "Absolute compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaclt[_n_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facgt))]
pub fn svaclt_n_f32(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svbool_t {
    svaclt_f32(pg, op1, svdup_n_f32(op2))
}
#[doc = "Absolute compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaclt[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facgt))]
pub fn svaclt_f64(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool_t {
    svacgt_f64(pg, op2, op1)
}
#[doc = "Absolute compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaclt[_n_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(facgt))]
pub fn svaclt_n_f64(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svbool_t {
    svaclt_f64(pg, op1, svdup_n_f64(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fadd))]
pub fn svadd_f32_m(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fadd.nxv4f32")]
        fn _svadd_f32_m(pg: svbool4_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t;
    }
    unsafe { _svadd_f32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fadd))]
pub fn svadd_n_f32_m(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svadd_f32_m(pg, op1, svdup_n_f32(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fadd))]
pub fn svadd_f32_x(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    svadd_f32_m(pg, op1, op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fadd))]
pub fn svadd_n_f32_x(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svadd_f32_x(pg, op1, svdup_n_f32(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fadd))]
pub fn svadd_f32_z(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    svadd_f32_m(pg, svsel_f32(pg, op1, svdup_n_f32(0.0)), op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fadd))]
pub fn svadd_n_f32_z(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svadd_f32_z(pg, op1, svdup_n_f32(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fadd))]
pub fn svadd_f64_m(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fadd.nxv2f64")]
        fn _svadd_f64_m(pg: svbool2_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t;
    }
    unsafe { _svadd_f64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fadd))]
pub fn svadd_n_f64_m(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svadd_f64_m(pg, op1, svdup_n_f64(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fadd))]
pub fn svadd_f64_x(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    svadd_f64_m(pg, op1, op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fadd))]
pub fn svadd_n_f64_x(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svadd_f64_x(pg, op1, svdup_n_f64(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fadd))]
pub fn svadd_f64_z(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    svadd_f64_m(pg, svsel_f64(pg, op1, svdup_n_f64(0.0)), op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fadd))]
pub fn svadd_n_f64_z(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svadd_f64_z(pg, op1, svdup_n_f64(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.add.nxv16i8")]
        fn _svadd_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t;
    }
    unsafe { _svadd_s8_m(pg, op1, op2) }
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_s8_m(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svadd_s8_m(pg, op1, svdup_n_s8(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_s8_x(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svadd_s8_m(pg, op1, op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_s8_x(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svadd_s8_x(pg, op1, svdup_n_s8(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_s8_z(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svadd_s8_m(pg, svsel_s8(pg, op1, svdup_n_s8(0)), op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_s8_z(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svadd_s8_z(pg, op1, svdup_n_s8(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_s16_m(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.add.nxv8i16")]
        fn _svadd_s16_m(pg: svbool8_t, op1: svint16_t, op2: svint16_t) -> svint16_t;
    }
    unsafe { _svadd_s16_m(simd_cast(pg), op1, op2) }
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_s16_m(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svadd_s16_m(pg, op1, svdup_n_s16(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_s16_x(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svadd_s16_m(pg, op1, op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_s16_x(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svadd_s16_x(pg, op1, svdup_n_s16(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_s16_z(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svadd_s16_m(pg, svsel_s16(pg, op1, svdup_n_s16(0)), op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_s16_z(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svadd_s16_z(pg, op1, svdup_n_s16(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_s32_m(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.add.nxv4i32")]
        fn _svadd_s32_m(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svint32_t;
    }
    unsafe { _svadd_s32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_s32_m(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svadd_s32_m(pg, op1, svdup_n_s32(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_s32_x(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svadd_s32_m(pg, op1, op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_s32_x(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svadd_s32_x(pg, op1, svdup_n_s32(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_s32_z(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svadd_s32_m(pg, svsel_s32(pg, op1, svdup_n_s32(0)), op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_s32_z(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svadd_s32_z(pg, op1, svdup_n_s32(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_s64_m(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.add.nxv2i64")]
        fn _svadd_s64_m(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svint64_t;
    }
    unsafe { _svadd_s64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_s64_m(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svadd_s64_m(pg, op1, svdup_n_s64(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_s64_x(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svadd_s64_m(pg, op1, op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_s64_x(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svadd_s64_x(pg, op1, svdup_n_s64(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_s64_z(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svadd_s64_m(pg, svsel_s64(pg, op1, svdup_n_s64(0)), op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_s64_z(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svadd_s64_z(pg, op1, svdup_n_s64(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_u8_m(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    unsafe { svadd_s8_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_u8_m(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svadd_u8_m(pg, op1, svdup_n_u8(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_u8_x(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svadd_u8_m(pg, op1, op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_u8_x(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svadd_u8_x(pg, op1, svdup_n_u8(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_u8_z(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svadd_u8_m(pg, svsel_u8(pg, op1, svdup_n_u8(0)), op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_u8_z(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svadd_u8_z(pg, op1, svdup_n_u8(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_u16_m(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    unsafe { svadd_s16_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_u16_m(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svadd_u16_m(pg, op1, svdup_n_u16(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_u16_x(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svadd_u16_m(pg, op1, op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_u16_x(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svadd_u16_x(pg, op1, svdup_n_u16(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_u16_z(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svadd_u16_m(pg, svsel_u16(pg, op1, svdup_n_u16(0)), op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_u16_z(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svadd_u16_z(pg, op1, svdup_n_u16(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_u32_m(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    unsafe { svadd_s32_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_u32_m(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svadd_u32_m(pg, op1, svdup_n_u32(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_u32_x(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svadd_u32_m(pg, op1, op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_u32_x(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svadd_u32_x(pg, op1, svdup_n_u32(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_u32_z(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svadd_u32_m(pg, svsel_u32(pg, op1, svdup_n_u32(0)), op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_u32_z(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svadd_u32_z(pg, op1, svdup_n_u32(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_u64_m(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    unsafe { svadd_s64_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_u64_m(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svadd_u64_m(pg, op1, svdup_n_u64(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_u64_x(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svadd_u64_m(pg, op1, op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_u64_x(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svadd_u64_x(pg, op1, svdup_n_u64(op2))
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_u64_z(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svadd_u64_m(pg, svsel_u64(pg, op1, svdup_n_u64(0)), op2)
}
#[doc = "Add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadd[_n_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(add))]
pub fn svadd_n_u64_z(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svadd_u64_z(pg, op1, svdup_n_u64(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.and.nxv16i8")]
        fn _svand_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t;
    }
    unsafe { _svand_s8_m(pg, op1, op2) }
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_s8_m(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svand_s8_m(pg, op1, svdup_n_s8(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_s8_x(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svand_s8_m(pg, op1, op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_s8_x(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svand_s8_x(pg, op1, svdup_n_s8(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_s8_z(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svand_s8_m(pg, svsel_s8(pg, op1, svdup_n_s8(0)), op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_s8_z(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svand_s8_z(pg, op1, svdup_n_s8(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_s16_m(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.and.nxv8i16")]
        fn _svand_s16_m(pg: svbool8_t, op1: svint16_t, op2: svint16_t) -> svint16_t;
    }
    unsafe { _svand_s16_m(simd_cast(pg), op1, op2) }
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_s16_m(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svand_s16_m(pg, op1, svdup_n_s16(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_s16_x(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svand_s16_m(pg, op1, op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_s16_x(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svand_s16_x(pg, op1, svdup_n_s16(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_s16_z(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svand_s16_m(pg, svsel_s16(pg, op1, svdup_n_s16(0)), op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_s16_z(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svand_s16_z(pg, op1, svdup_n_s16(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_s32_m(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.and.nxv4i32")]
        fn _svand_s32_m(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svint32_t;
    }
    unsafe { _svand_s32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_s32_m(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svand_s32_m(pg, op1, svdup_n_s32(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_s32_x(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svand_s32_m(pg, op1, op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_s32_x(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svand_s32_x(pg, op1, svdup_n_s32(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_s32_z(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svand_s32_m(pg, svsel_s32(pg, op1, svdup_n_s32(0)), op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_s32_z(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svand_s32_z(pg, op1, svdup_n_s32(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_s64_m(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.and.nxv2i64")]
        fn _svand_s64_m(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svint64_t;
    }
    unsafe { _svand_s64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_s64_m(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svand_s64_m(pg, op1, svdup_n_s64(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_s64_x(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svand_s64_m(pg, op1, op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_s64_x(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svand_s64_x(pg, op1, svdup_n_s64(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_s64_z(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svand_s64_m(pg, svsel_s64(pg, op1, svdup_n_s64(0)), op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_s64_z(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svand_s64_z(pg, op1, svdup_n_s64(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_u8_m(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    unsafe { svand_s8_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_u8_m(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svand_u8_m(pg, op1, svdup_n_u8(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_u8_x(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svand_u8_m(pg, op1, op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_u8_x(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svand_u8_x(pg, op1, svdup_n_u8(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_u8_z(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svand_u8_m(pg, svsel_u8(pg, op1, svdup_n_u8(0)), op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_u8_z(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svand_u8_z(pg, op1, svdup_n_u8(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_u16_m(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    unsafe { svand_s16_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_u16_m(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svand_u16_m(pg, op1, svdup_n_u16(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_u16_x(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svand_u16_m(pg, op1, op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_u16_x(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svand_u16_x(pg, op1, svdup_n_u16(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_u16_z(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svand_u16_m(pg, svsel_u16(pg, op1, svdup_n_u16(0)), op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_u16_z(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svand_u16_z(pg, op1, svdup_n_u16(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_u32_m(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    unsafe { svand_s32_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_u32_m(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svand_u32_m(pg, op1, svdup_n_u32(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_u32_x(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svand_u32_m(pg, op1, op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_u32_x(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svand_u32_x(pg, op1, svdup_n_u32(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_u32_z(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svand_u32_m(pg, svsel_u32(pg, op1, svdup_n_u32(0)), op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_u32_z(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svand_u32_z(pg, op1, svdup_n_u32(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_u64_m(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    unsafe { svand_s64_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_u64_m(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svand_u64_m(pg, op1, svdup_n_u64(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_u64_x(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svand_u64_m(pg, op1, op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_u64_x(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svand_u64_x(pg, op1, svdup_n_u64(op2))
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_u64_z(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svand_u64_m(pg, svsel_u64(pg, op1, svdup_n_u64(0)), op2)
}
#[doc = "Bitwise AND"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svand[_n_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(and))]
pub fn svand_n_u64_z(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svand_u64_z(pg, op1, svdup_n_u64(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_b]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_b_z(pg: svbool_t, op1: svbool_t, op2: svbool_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bic.z.nvx16i1")]
        fn _svbic_b_z(pg: svbool_t, op1: svbool_t, op2: svbool_t) -> svbool_t;
    }
    unsafe { _svbic_b_z(pg, op1, op2) }
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bic.nxv16i8")]
        fn _svbic_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t;
    }
    unsafe { _svbic_s8_m(pg, op1, op2) }
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_s8_m(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svbic_s8_m(pg, op1, svdup_n_s8(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_s8_x(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svbic_s8_m(pg, op1, op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_s8_x(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svbic_s8_x(pg, op1, svdup_n_s8(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_s8_z(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svbic_s8_m(pg, svsel_s8(pg, op1, svdup_n_s8(0)), op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_s8_z(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svbic_s8_z(pg, op1, svdup_n_s8(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_s16_m(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bic.nxv8i16")]
        fn _svbic_s16_m(pg: svbool8_t, op1: svint16_t, op2: svint16_t) -> svint16_t;
    }
    unsafe { _svbic_s16_m(simd_cast(pg), op1, op2) }
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_s16_m(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svbic_s16_m(pg, op1, svdup_n_s16(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_s16_x(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svbic_s16_m(pg, op1, op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_s16_x(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svbic_s16_x(pg, op1, svdup_n_s16(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_s16_z(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svbic_s16_m(pg, svsel_s16(pg, op1, svdup_n_s16(0)), op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_s16_z(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svbic_s16_z(pg, op1, svdup_n_s16(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_s32_m(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bic.nxv4i32")]
        fn _svbic_s32_m(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svint32_t;
    }
    unsafe { _svbic_s32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_s32_m(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svbic_s32_m(pg, op1, svdup_n_s32(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_s32_x(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svbic_s32_m(pg, op1, op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_s32_x(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svbic_s32_x(pg, op1, svdup_n_s32(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_s32_z(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svbic_s32_m(pg, svsel_s32(pg, op1, svdup_n_s32(0)), op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_s32_z(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svbic_s32_z(pg, op1, svdup_n_s32(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_s64_m(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bic.nxv2i64")]
        fn _svbic_s64_m(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svint64_t;
    }
    unsafe { _svbic_s64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_s64_m(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svbic_s64_m(pg, op1, svdup_n_s64(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_s64_x(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svbic_s64_m(pg, op1, op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_s64_x(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svbic_s64_x(pg, op1, svdup_n_s64(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_s64_z(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svbic_s64_m(pg, svsel_s64(pg, op1, svdup_n_s64(0)), op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_s64_z(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svbic_s64_z(pg, op1, svdup_n_s64(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_u8_m(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    unsafe { svbic_s8_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_u8_m(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svbic_u8_m(pg, op1, svdup_n_u8(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_u8_x(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svbic_u8_m(pg, op1, op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_u8_x(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svbic_u8_x(pg, op1, svdup_n_u8(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_u8_z(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svbic_u8_m(pg, svsel_u8(pg, op1, svdup_n_u8(0)), op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_u8_z(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svbic_u8_z(pg, op1, svdup_n_u8(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_u16_m(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    unsafe { svbic_s16_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_u16_m(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svbic_u16_m(pg, op1, svdup_n_u16(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_u16_x(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svbic_u16_m(pg, op1, op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_u16_x(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svbic_u16_x(pg, op1, svdup_n_u16(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_u16_z(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svbic_u16_m(pg, svsel_u16(pg, op1, svdup_n_u16(0)), op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_u16_z(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svbic_u16_z(pg, op1, svdup_n_u16(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_u32_m(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    unsafe { svbic_s32_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_u32_m(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svbic_u32_m(pg, op1, svdup_n_u32(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_u32_x(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svbic_u32_m(pg, op1, op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_u32_x(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svbic_u32_x(pg, op1, svdup_n_u32(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_u32_z(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svbic_u32_m(pg, svsel_u32(pg, op1, svdup_n_u32(0)), op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_u32_z(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svbic_u32_z(pg, op1, svdup_n_u32(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_u64_m(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    unsafe { svbic_s64_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_u64_m(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svbic_u64_m(pg, op1, svdup_n_u64(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_u64_x(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svbic_u64_m(pg, op1, op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_u64_x(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svbic_u64_x(pg, op1, svdup_n_u64(op2))
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_u64_z(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svbic_u64_m(pg, svsel_u64(pg, op1, svdup_n_u64(0)), op2)
}
#[doc = "Bitwise clear"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbic[_n_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(bic))]
pub fn svbic_n_u64_z(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svbic_u64_z(pg, op1, svdup_n_u64(op2))
}
#[doc = "Break after first true condition"]
#[doc = ""]
#[doc = "Break after first true condition"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbrka[_b]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(brka))]
pub fn svbrka_b_m(inactive: svbool_t, pg: svbool_t, op: svbool_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.brka.nxv16i1")]
        fn _svbrka_b_m(inactive: svbool_t, pg: svbool_t, op: svbool_t) -> svbool_t;
    }
    unsafe { _svbrka_b_m(inactive, pg, op) }
}
#[doc = "Break after first true condition"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbrka[_b]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(brka))]
pub fn svbrka_b_z(pg: svbool_t, op: svbool_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.brka.z.nxv16i1")]
        fn _svbrka_b_z(pg: svbool_t, op: svbool_t) -> svbool_t;
    }
    unsafe { _svbrka_b_z(pg, op) }
}
#[doc = "Break before first true condition"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbrkb[_b]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(brkb))]
pub fn svbrkb_b_m(inactive: svbool_t, pg: svbool_t, op: svbool_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.brkb.nxv16i1")]
        fn _svbrkb_b_m(inactive: svbool_t, pg: svbool_t, op: svbool_t) -> svbool_t;
    }
    unsafe { _svbrkb_b_m(inactive, pg, op) }
}
#[doc = "Break before first true condition"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbrkb[_b]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(brkb))]
pub fn svbrkb_b_z(pg: svbool_t, op: svbool_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.brkb.z.nxv16i1")]
        fn _svbrkb_b_z(pg: svbool_t, op: svbool_t) -> svbool_t;
    }
    unsafe { _svbrkb_b_z(pg, op) }
}
#[doc = "Propagate break to next partition"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbrkn[_b]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(brkn))]
pub fn svbrkn_b_z(pg: svbool_t, op1: svbool_t, op2: svbool_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.brkn.z.nxv16i1")]
        fn _svbrkn_b_z(pg: svbool_t, op1: svbool_t, op2: svbool_t) -> svbool_t;
    }
    unsafe { _svbrkn_b_z(pg, op1, op2) }
}
#[doc = "Break after first true condition, propagating from previous partition"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbrkpa[_b]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(brkpa))]
pub fn svbrkpa_b_z(pg: svbool_t, op1: svbool_t, op2: svbool_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.brkpa.z.nxv16i1"
        )]
        fn _svbrkpa_b_z(pg: svbool_t, op1: svbool_t, op2: svbool_t) -> svbool_t;
    }
    unsafe { _svbrkpa_b_z(pg, op1, op2) }
}
#[doc = "Break before first true condition, propagating from previous partition"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbrkpb[_b]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(brkpb))]
pub fn svbrkpb_b_z(pg: svbool_t, op1: svbool_t, op2: svbool_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.brkpb.z.nxv16i1"
        )]
        fn _svbrkpb_b_z(pg: svbool_t, op1: svbool_t, op2: svbool_t) -> svbool_t;
    }
    unsafe { _svbrkpb_b_z(pg, op1, op2) }
}
#[doc = "Complex add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcadd[_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcadd, IMM_ROTATION = 90))]
pub fn svcadd_f32_m<const IMM_ROTATION: i32>(
    pg: svbool_t,
    op1: svfloat32_t,
    op2: svfloat32_t,
) -> svfloat32_t {
    static_assert!(IMM_ROTATION == 90 || IMM_ROTATION == 270);
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcadd.nxv4f32")]
        fn _svcadd_f32_m(
            pg: svbool4_t,
            op1: svfloat32_t,
            op2: svfloat32_t,
            imm_rotation: i32,
        ) -> svfloat32_t;
    }
    unsafe { _svcadd_f32_m(simd_cast(pg), op1, op2, IMM_ROTATION) }
}
#[doc = "Complex add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcadd[_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcadd, IMM_ROTATION = 90))]
pub fn svcadd_f32_x<const IMM_ROTATION: i32>(
    pg: svbool_t,
    op1: svfloat32_t,
    op2: svfloat32_t,
) -> svfloat32_t {
    svcadd_f32_m::<IMM_ROTATION>(pg, op1, op2)
}
#[doc = "Complex add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcadd[_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcadd, IMM_ROTATION = 90))]
pub fn svcadd_f32_z<const IMM_ROTATION: i32>(
    pg: svbool_t,
    op1: svfloat32_t,
    op2: svfloat32_t,
) -> svfloat32_t {
    svcadd_f32_m::<IMM_ROTATION>(pg, svsel_f32(pg, op1, svdup_n_f32(0.0)), op2)
}
#[doc = "Complex add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcadd[_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcadd, IMM_ROTATION = 90))]
pub fn svcadd_f64_m<const IMM_ROTATION: i32>(
    pg: svbool_t,
    op1: svfloat64_t,
    op2: svfloat64_t,
) -> svfloat64_t {
    static_assert!(IMM_ROTATION == 90 || IMM_ROTATION == 270);
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcadd.nxv2f64")]
        fn _svcadd_f64_m(
            pg: svbool2_t,
            op1: svfloat64_t,
            op2: svfloat64_t,
            imm_rotation: i32,
        ) -> svfloat64_t;
    }
    unsafe { _svcadd_f64_m(simd_cast(pg), op1, op2, IMM_ROTATION) }
}
#[doc = "Complex add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcadd[_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcadd, IMM_ROTATION = 90))]
pub fn svcadd_f64_x<const IMM_ROTATION: i32>(
    pg: svbool_t,
    op1: svfloat64_t,
    op2: svfloat64_t,
) -> svfloat64_t {
    svcadd_f64_m::<IMM_ROTATION>(pg, op1, op2)
}
#[doc = "Complex add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcadd[_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcadd, IMM_ROTATION = 90))]
pub fn svcadd_f64_z<const IMM_ROTATION: i32>(
    pg: svbool_t,
    op1: svfloat64_t,
    op2: svfloat64_t,
) -> svfloat64_t {
    svcadd_f64_m::<IMM_ROTATION>(pg, svsel_f64(pg, op1, svdup_n_f64(0.0)), op2)
}
#[doc = "Complex multiply-add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmla[_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmla, IMM_ROTATION = 90))]
pub fn svcmla_f32_m<const IMM_ROTATION: i32>(
    pg: svbool_t,
    op1: svfloat32_t,
    op2: svfloat32_t,
    op3: svfloat32_t,
) -> svfloat32_t {
    static_assert!(
        IMM_ROTATION == 0 || IMM_ROTATION == 90 || IMM_ROTATION == 180 || IMM_ROTATION == 270
    );
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcmla.nxv4f32")]
        fn _svcmla_f32_m(
            pg: svbool4_t,
            op1: svfloat32_t,
            op2: svfloat32_t,
            op3: svfloat32_t,
            imm_rotation: i32,
        ) -> svfloat32_t;
    }
    unsafe { _svcmla_f32_m(simd_cast(pg), op1, op2, op3, IMM_ROTATION) }
}
#[doc = "Complex multiply-add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmla[_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmla, IMM_ROTATION = 90))]
pub fn svcmla_f32_x<const IMM_ROTATION: i32>(
    pg: svbool_t,
    op1: svfloat32_t,
    op2: svfloat32_t,
    op3: svfloat32_t,
) -> svfloat32_t {
    svcmla_f32_m::<IMM_ROTATION>(pg, op1, op2, op3)
}
#[doc = "Complex multiply-add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmla[_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmla, IMM_ROTATION = 90))]
pub fn svcmla_f32_z<const IMM_ROTATION: i32>(
    pg: svbool_t,
    op1: svfloat32_t,
    op2: svfloat32_t,
    op3: svfloat32_t,
) -> svfloat32_t {
    svcmla_f32_m::<IMM_ROTATION>(pg, svsel_f32(pg, op1, svdup_n_f32(0.0)), op2, op3)
}
#[doc = "Complex multiply-add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmla[_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmla, IMM_ROTATION = 90))]
pub fn svcmla_f64_m<const IMM_ROTATION: i32>(
    pg: svbool_t,
    op1: svfloat64_t,
    op2: svfloat64_t,
    op3: svfloat64_t,
) -> svfloat64_t {
    static_assert!(
        IMM_ROTATION == 0 || IMM_ROTATION == 90 || IMM_ROTATION == 180 || IMM_ROTATION == 270
    );
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcmla.nxv2f64")]
        fn _svcmla_f64_m(
            pg: svbool2_t,
            op1: svfloat64_t,
            op2: svfloat64_t,
            op3: svfloat64_t,
            imm_rotation: i32,
        ) -> svfloat64_t;
    }
    unsafe { _svcmla_f64_m(simd_cast(pg), op1, op2, op3, IMM_ROTATION) }
}
#[doc = "Complex multiply-add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmla[_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmla, IMM_ROTATION = 90))]
pub fn svcmla_f64_x<const IMM_ROTATION: i32>(
    pg: svbool_t,
    op1: svfloat64_t,
    op2: svfloat64_t,
    op3: svfloat64_t,
) -> svfloat64_t {
    svcmla_f64_m::<IMM_ROTATION>(pg, op1, op2, op3)
}
#[doc = "Complex multiply-add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmla[_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmla, IMM_ROTATION = 90))]
pub fn svcmla_f64_z<const IMM_ROTATION: i32>(
    pg: svbool_t,
    op1: svfloat64_t,
    op2: svfloat64_t,
    op3: svfloat64_t,
) -> svfloat64_t {
    svcmla_f64_m::<IMM_ROTATION>(pg, svsel_f64(pg, op1, svdup_n_f64(0.0)), op2, op3)
}
#[doc = "Complex multiply-add with rotate"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmla_lane[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmla, IMM_INDEX = 0, IMM_ROTATION = 90))]
pub fn svcmla_lane_f32<const IMM_INDEX: i32, const IMM_ROTATION: i32>(
    op1: svfloat32_t,
    op2: svfloat32_t,
    op3: svfloat32_t,
) -> svfloat32_t {
    static_assert_range!(IMM_INDEX, 0, 1);
    static_assert!(
        IMM_ROTATION == 0 || IMM_ROTATION == 90 || IMM_ROTATION == 180 || IMM_ROTATION == 270
    );
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.fcmla.lane.x.nxv4f32"
        )]
        fn _svcmla_lane_f32(
            op1: svfloat32_t,
            op2: svfloat32_t,
            op3: svfloat32_t,
            imm_index: i32,
            imm_rotation: i32,
        ) -> svfloat32_t;
    }
    unsafe { _svcmla_lane_f32(op1, op2, op3, IMM_INDEX, IMM_ROTATION) }
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmeq))]
pub fn svcmpeq_f32(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcmpeq.nxv4f32")]
        fn _svcmpeq_f32(pg: svbool4_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svcmpeq_f32(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_n_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmeq))]
pub fn svcmpeq_n_f32(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svbool_t {
    svcmpeq_f32(pg, op1, svdup_n_f32(op2))
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmeq))]
pub fn svcmpeq_f64(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcmpeq.nxv2f64")]
        fn _svcmpeq_f64(pg: svbool2_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool2_t;
    }
    unsafe { simd_cast(_svcmpeq_f64(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_n_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmeq))]
pub fn svcmpeq_n_f64(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svbool_t {
    svcmpeq_f64(pg, op1, svdup_n_f64(op2))
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpeq.nxv16i8")]
        fn _svcmpeq_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svbool_t;
    }
    unsafe { _svcmpeq_s8(pg, op1, op2) }
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_n_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_n_s8(pg: svbool_t, op1: svint8_t, op2: i8) -> svbool_t {
    svcmpeq_s8(pg, op1, svdup_n_s8(op2))
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_s16(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpeq.nxv8i16")]
        fn _svcmpeq_s16(pg: svbool8_t, op1: svint16_t, op2: svint16_t) -> svbool8_t;
    }
    unsafe { simd_cast(_svcmpeq_s16(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_n_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_n_s16(pg: svbool_t, op1: svint16_t, op2: i16) -> svbool_t {
    svcmpeq_s16(pg, op1, svdup_n_s16(op2))
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_s32(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpeq.nxv4i32")]
        fn _svcmpeq_s32(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svcmpeq_s32(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_n_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_n_s32(pg: svbool_t, op1: svint32_t, op2: i32) -> svbool_t {
    svcmpeq_s32(pg, op1, svdup_n_s32(op2))
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_s64(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpeq.nxv2i64")]
        fn _svcmpeq_s64(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svbool2_t;
    }
    unsafe { simd_cast(_svcmpeq_s64(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_n_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_n_s64(pg: svbool_t, op1: svint64_t, op2: i64) -> svbool_t {
    svcmpeq_s64(pg, op1, svdup_n_s64(op2))
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_u8(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svbool_t {
    unsafe { svcmpeq_s8(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_n_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_n_u8(pg: svbool_t, op1: svuint8_t, op2: u8) -> svbool_t {
    svcmpeq_u8(pg, op1, svdup_n_u8(op2))
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_u16(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svbool_t {
    unsafe { svcmpeq_s16(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_n_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_n_u16(pg: svbool_t, op1: svuint16_t, op2: u16) -> svbool_t {
    svcmpeq_u16(pg, op1, svdup_n_u16(op2))
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_u32(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svbool_t {
    unsafe { svcmpeq_s32(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_n_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_n_u32(pg: svbool_t, op1: svuint32_t, op2: u32) -> svbool_t {
    svcmpeq_u32(pg, op1, svdup_n_u32(op2))
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_u64(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svbool_t {
    unsafe { svcmpeq_s64(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq[_n_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpeq))]
pub fn svcmpeq_n_u64(pg: svbool_t, op1: svuint64_t, op2: u64) -> svbool_t {
    svcmpeq_u64(pg, op1, svdup_n_u64(op2))
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub fn svcmpgt_f32(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcmpgt.nxv4f32")]
        fn _svcmpgt_f32(pg: svbool4_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svcmpgt_f32(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_n_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub fn svcmpgt_n_f32(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svbool_t {
    svcmpgt_f32(pg, op1, svdup_n_f32(op2))
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub fn svcmpgt_f64(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcmpgt.nxv2f64")]
        fn _svcmpgt_f64(pg: svbool2_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool2_t;
    }
    unsafe { simd_cast(_svcmpgt_f64(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_n_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub fn svcmpgt_n_f64(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svbool_t {
    svcmpgt_f64(pg, op1, svdup_n_f64(op2))
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmpgt_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpgt.nxv16i8")]
        fn _svcmpgt_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svbool_t;
    }
    unsafe { _svcmpgt_s8(pg, op1, op2) }
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_n_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmpgt_n_s8(pg: svbool_t, op1: svint8_t, op2: i8) -> svbool_t {
    svcmpgt_s8(pg, op1, svdup_n_s8(op2))
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmpgt_s16(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpgt.nxv8i16")]
        fn _svcmpgt_s16(pg: svbool8_t, op1: svint16_t, op2: svint16_t) -> svbool8_t;
    }
    unsafe { simd_cast(_svcmpgt_s16(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_n_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmpgt_n_s16(pg: svbool_t, op1: svint16_t, op2: i16) -> svbool_t {
    svcmpgt_s16(pg, op1, svdup_n_s16(op2))
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmpgt_s32(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpgt.nxv4i32")]
        fn _svcmpgt_s32(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svcmpgt_s32(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_n_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmpgt_n_s32(pg: svbool_t, op1: svint32_t, op2: i32) -> svbool_t {
    svcmpgt_s32(pg, op1, svdup_n_s32(op2))
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmpgt_s64(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpgt.nxv2i64")]
        fn _svcmpgt_s64(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svbool2_t;
    }
    unsafe { simd_cast(_svcmpgt_s64(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_n_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmpgt_n_s64(pg: svbool_t, op1: svint64_t, op2: i64) -> svbool_t {
    svcmpgt_s64(pg, op1, svdup_n_s64(op2))
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmpgt_u8(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svbool_t {
    unsafe { svcmpgt_s8(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_n_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmpgt_n_u8(pg: svbool_t, op1: svuint8_t, op2: u8) -> svbool_t {
    svcmpgt_u8(pg, op1, svdup_n_u8(op2))
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmpgt_u16(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svbool_t {
    unsafe { svcmpgt_s16(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_n_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmpgt_n_u16(pg: svbool_t, op1: svuint16_t, op2: u16) -> svbool_t {
    svcmpgt_u16(pg, op1, svdup_n_u16(op2))
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmpgt_u32(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svbool_t {
    unsafe { svcmpgt_s32(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_n_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmpgt_n_u32(pg: svbool_t, op1: svuint32_t, op2: u32) -> svbool_t {
    svcmpgt_u32(pg, op1, svdup_n_u32(op2))
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmpgt_u64(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svbool_t {
    unsafe { svcmpgt_s64(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare greater than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpgt[_n_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmpgt_n_u64(pg: svbool_t, op1: svuint64_t, op2: u64) -> svbool_t {
    svcmpgt_u64(pg, op1, svdup_n_u64(op2))
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub fn svcmplt_f32(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool_t {
    svcmpgt_f32(pg, op2, op1)
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_n_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub fn svcmplt_n_f32(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svbool_t {
    svcmplt_f32(pg, op1, svdup_n_f32(op2))
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub fn svcmplt_f64(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool_t {
    svcmpgt_f64(pg, op2, op1)
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_n_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmgt))]
pub fn svcmplt_n_f64(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svbool_t {
    svcmplt_f64(pg, op1, svdup_n_f64(op2))
}

#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmne))]
pub fn svcmpne_f32(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcmpne.nxv4f32")]
        fn _svcmpne_f32(pg: svbool4_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svcmpne_f32(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_n_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmne))]
pub fn svcmpne_n_f32(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svbool_t {
    svcmpne_f32(pg, op1, svdup_n_f32(op2))
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmne))]
pub fn svcmpne_f64(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcmpne.nxv2f64")]
        fn _svcmpne_f64(pg: svbool2_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool2_t;
    }
    unsafe { simd_cast(_svcmpne_f64(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_n_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmne))]
pub fn svcmpne_n_f64(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svbool_t {
    svcmpne_f64(pg, op1, svdup_n_f64(op2))
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpne.nxv16i8")]
        fn _svcmpne_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svbool_t;
    }
    unsafe { _svcmpne_s8(pg, op1, op2) }
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_n_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_n_s8(pg: svbool_t, op1: svint8_t, op2: i8) -> svbool_t {
    svcmpne_s8(pg, op1, svdup_n_s8(op2))
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_s16(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpne.nxv8i16")]
        fn _svcmpne_s16(pg: svbool8_t, op1: svint16_t, op2: svint16_t) -> svbool8_t;
    }
    unsafe { simd_cast(_svcmpne_s16(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_n_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_n_s16(pg: svbool_t, op1: svint16_t, op2: i16) -> svbool_t {
    svcmpne_s16(pg, op1, svdup_n_s16(op2))
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_s32(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpne.nxv4i32")]
        fn _svcmpne_s32(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svcmpne_s32(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_n_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_n_s32(pg: svbool_t, op1: svint32_t, op2: i32) -> svbool_t {
    svcmpne_s32(pg, op1, svdup_n_s32(op2))
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_s64(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpne.nxv2i64")]
        fn _svcmpne_s64(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svbool2_t;
    }
    unsafe { simd_cast(_svcmpne_s64(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_n_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_n_s64(pg: svbool_t, op1: svint64_t, op2: i64) -> svbool_t {
    svcmpne_s64(pg, op1, svdup_n_s64(op2))
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_u8(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svbool_t {
    unsafe { svcmpne_s8(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_n_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_n_u8(pg: svbool_t, op1: svuint8_t, op2: u8) -> svbool_t {
    svcmpne_u8(pg, op1, svdup_n_u8(op2))
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_u16(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svbool_t {
    unsafe { svcmpne_s16(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_n_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_n_u16(pg: svbool_t, op1: svuint16_t, op2: u16) -> svbool_t {
    svcmpne_u16(pg, op1, svdup_n_u16(op2))
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_u32(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svbool_t {
    unsafe { svcmpne_s32(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_n_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_n_u32(pg: svbool_t, op1: svuint32_t, op2: u32) -> svbool_t {
    svcmpne_u32(pg, op1, svdup_n_u32(op2))
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_u64(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svbool_t {
    unsafe { svcmpne_s64(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare not equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpne[_n_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpne))]
pub fn svcmpne_n_u64(pg: svbool_t, op1: svuint64_t, op2: u64) -> svbool_t {
    svcmpne_u64(pg, op1, svdup_n_u64(op2))
}

#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmge))]
pub fn svcmpge_f32(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcmpge.nxv4f32")]
        fn _svcmpge_f32(pg: svbool4_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svcmpge_f32(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_n_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmge))]
pub fn svcmpge_n_f32(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svbool_t {
    svcmpge_f32(pg, op1, svdup_n_f32(op2))
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmge))]
pub fn svcmpge_f64(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcmpge.nxv2f64")]
        fn _svcmpge_f64(pg: svbool2_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool2_t;
    }
    unsafe { simd_cast(_svcmpge_f64(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_n_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmge))]
pub fn svcmpge_n_f64(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svbool_t {
    svcmpge_f64(pg, op1, svdup_n_f64(op2))
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpge))]
pub fn svcmpge_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpge.nxv16i8")]
        fn _svcmpge_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svbool_t;
    }
    unsafe { _svcmpge_s8(pg, op1, op2) }
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_n_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpge))]
pub fn svcmpge_n_s8(pg: svbool_t, op1: svint8_t, op2: i8) -> svbool_t {
    svcmpge_s8(pg, op1, svdup_n_s8(op2))
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpge))]
pub fn svcmpge_s16(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpge.nxv8i16")]
        fn _svcmpge_s16(pg: svbool8_t, op1: svint16_t, op2: svint16_t) -> svbool8_t;
    }
    unsafe { simd_cast(_svcmpge_s16(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_n_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpge))]
pub fn svcmpge_n_s16(pg: svbool_t, op1: svint16_t, op2: i16) -> svbool_t {
    svcmpge_s16(pg, op1, svdup_n_s16(op2))
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpge))]
pub fn svcmpge_s32(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpge.nxv4i32")]
        fn _svcmpge_s32(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svcmpge_s32(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_n_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpge))]
pub fn svcmpge_n_s32(pg: svbool_t, op1: svint32_t, op2: i32) -> svbool_t {
    svcmpge_s32(pg, op1, svdup_n_s32(op2))
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpge))]
pub fn svcmpge_s64(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmpge.nxv2i64")]
        fn _svcmpge_s64(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svbool2_t;
    }
    unsafe { simd_cast(_svcmpge_s64(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_n_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpge))]
pub fn svcmpge_n_s64(pg: svbool_t, op1: svint64_t, op2: i64) -> svbool_t {
    svcmpge_s64(pg, op1, svdup_n_s64(op2))
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmhs))]
pub fn svcmpge_u8(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svbool_t {
    unsafe { svcmpge_s8(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_n_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmhs))]
pub fn svcmpge_n_u8(pg: svbool_t, op1: svuint8_t, op2: u8) -> svbool_t {
    svcmpge_u8(pg, op1, svdup_n_u8(op2))
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmhs))]
pub fn svcmpge_u16(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svbool_t {
    unsafe { svcmpge_s16(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_n_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmhs))]
pub fn svcmpge_n_u16(pg: svbool_t, op1: svuint16_t, op2: u16) -> svbool_t {
    svcmpge_u16(pg, op1, svdup_n_u16(op2))
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmhs))]
pub fn svcmpge_u32(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svbool_t {
    unsafe { svcmpge_s32(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_n_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmhs))]
pub fn svcmpge_n_u32(pg: svbool_t, op1: svuint32_t, op2: u32) -> svbool_t {
    svcmpge_u32(pg, op1, svdup_n_u32(op2))
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmhs))]
pub fn svcmpge_u64(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svbool_t {
    unsafe { svcmpge_s64(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare greater than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpge[_n_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmhs))]
pub fn svcmpge_n_u64(pg: svbool_t, op1: svuint64_t, op2: u64) -> svbool_t {
    svcmpge_u64(pg, op1, svdup_n_u64(op2))
}

#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmle))]
pub fn svcmple_f32(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcmple.nxv4f32")]
        fn _svcmple_f32(pg: svbool4_t, op1: svfloat32_t, op2: svfloat32_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svcmple_f32(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_n_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmle))]
pub fn svcmple_n_f32(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svbool_t {
    svcmple_f32(pg, op1, svdup_n_f32(op2))
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmle))]
pub fn svcmple_f64(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fcmple.nxv2f64")]
        fn _svcmple_f64(pg: svbool2_t, op1: svfloat64_t, op2: svfloat64_t) -> svbool2_t;
    }
    unsafe { simd_cast(_svcmple_f64(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_n_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fcmle))]
pub fn svcmple_n_f64(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svbool_t {
    svcmple_f64(pg, op1, svdup_n_f64(op2))
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmple))]
pub fn svcmple_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmple.nxv16i8")]
        fn _svcmple_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svbool_t;
    }
    unsafe { _svcmple_s8(pg, op1, op2) }
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_n_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmple))]
pub fn svcmple_n_s8(pg: svbool_t, op1: svint8_t, op2: i8) -> svbool_t {
    svcmple_s8(pg, op1, svdup_n_s8(op2))
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmple))]
pub fn svcmple_s16(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmple.nxv8i16")]
        fn _svcmple_s16(pg: svbool8_t, op1: svint16_t, op2: svint16_t) -> svbool8_t;
    }
    unsafe { simd_cast(_svcmple_s16(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_n_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmple))]
pub fn svcmple_n_s16(pg: svbool_t, op1: svint16_t, op2: i16) -> svbool_t {
    svcmple_s16(pg, op1, svdup_n_s16(op2))
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmple))]
pub fn svcmple_s32(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmple.nxv4i32")]
        fn _svcmple_s32(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svcmple_s32(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_n_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmple))]
pub fn svcmple_n_s32(pg: svbool_t, op1: svint32_t, op2: i32) -> svbool_t {
    svcmple_s32(pg, op1, svdup_n_s32(op2))
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmple))]
pub fn svcmple_s64(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cmple.nxv2i64")]
        fn _svcmple_s64(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svbool2_t;
    }
    unsafe { simd_cast(_svcmple_s64(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_n_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmple))]
pub fn svcmple_n_s64(pg: svbool_t, op1: svint64_t, op2: i64) -> svbool_t {
    svcmple_s64(pg, op1, svdup_n_s64(op2))
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmls))]
pub fn svcmple_u8(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svbool_t {
    unsafe { svcmple_s8(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_n_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmls))]
pub fn svcmple_n_u8(pg: svbool_t, op1: svuint8_t, op2: u8) -> svbool_t {
    svcmple_u8(pg, op1, svdup_n_u8(op2))
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmls))]
pub fn svcmple_u16(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svbool_t {
    unsafe { svcmple_s16(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_n_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmls))]
pub fn svcmple_n_u16(pg: svbool_t, op1: svuint16_t, op2: u16) -> svbool_t {
    svcmple_u16(pg, op1, svdup_n_u16(op2))
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmls))]
pub fn svcmple_u32(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svbool_t {
    unsafe { svcmple_s32(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_n_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmls))]
pub fn svcmple_n_u32(pg: svbool_t, op1: svuint32_t, op2: u32) -> svbool_t {
    svcmple_u32(pg, op1, svdup_n_u32(op2))
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmls))]
pub fn svcmple_u64(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svbool_t {
    unsafe { svcmple_s64(pg, op1.as_signed(), op2.as_signed()) }
}
#[doc = "Compare less than or equal to"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmple[_n_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmls))]
pub fn svcmple_n_u64(pg: svbool_t, op1: svuint64_t, op2: u64) -> svbool_t {
    svcmple_u64(pg, op1, svdup_n_u64(op2))
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmplt_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svbool_t {
    svcmpgt_s8(pg, op2, op1)
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_n_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmplt_n_s8(pg: svbool_t, op1: svint8_t, op2: i8) -> svbool_t {
    svcmplt_s8(pg, op1, svdup_n_s8(op2))
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmplt_s16(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svbool_t {
    svcmpgt_s16(pg, op2, op1)
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_n_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmplt_n_s16(pg: svbool_t, op1: svint16_t, op2: i16) -> svbool_t {
    svcmplt_s16(pg, op1, svdup_n_s16(op2))
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmplt_s32(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svbool_t {
    svcmpgt_s32(pg, op2, op1)
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_n_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmplt_n_s32(pg: svbool_t, op1: svint32_t, op2: i32) -> svbool_t {
    svcmplt_s32(pg, op1, svdup_n_s32(op2))
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmplt_s64(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svbool_t {
    svcmpgt_s64(pg, op2, op1)
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_n_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmpgt))]
pub fn svcmplt_n_s64(pg: svbool_t, op1: svint64_t, op2: i64) -> svbool_t {
    svcmplt_s64(pg, op1, svdup_n_s64(op2))
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmplt_u8(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svbool_t {
    svcmpgt_u8(pg, op2, op1)
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_n_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmplt_n_u8(pg: svbool_t, op1: svuint8_t, op2: u8) -> svbool_t {
    svcmplt_u8(pg, op1, svdup_n_u8(op2))
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmplt_u16(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svbool_t {
    svcmpgt_u16(pg, op2, op1)
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_n_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmplt_n_u16(pg: svbool_t, op1: svuint16_t, op2: u16) -> svbool_t {
    svcmplt_u16(pg, op1, svdup_n_u16(op2))
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmplt_u32(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svbool_t {
    svcmpgt_u32(pg, op2, op1)
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_n_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmplt_n_u32(pg: svbool_t, op1: svuint32_t, op2: u32) -> svbool_t {
    svcmplt_u32(pg, op1, svdup_n_u32(op2))
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmplt_u64(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svbool_t {
    svcmpgt_u64(pg, op2, op1)
}
#[doc = "Compare less than"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmplt[_n_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cmphi))]
pub fn svcmplt_n_u64(pg: svbool_t, op1: svuint64_t, op2: u64) -> svbool_t {
    svcmplt_u64(pg, op1, svdup_n_u64(op2))
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f32[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(scvtf))]
pub fn svcvt_f32_s32_m(inactive: svfloat32_t, pg: svbool_t, op: svint32_t) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.scvtf.nxv4f32.nxv4i32"
        )]
        fn _svcvt_f32_s32_m(inactive: svfloat32_t, pg: svbool4_t, op: svint32_t) -> svfloat32_t;
    }
    unsafe { _svcvt_f32_s32_m(inactive, simd_cast(pg), op) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f32[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(scvtf))]
pub fn svcvt_f32_s32_x(pg: svbool_t, op: svint32_t) -> svfloat32_t {
    unsafe { svcvt_f32_s32_m(simd_reinterpret(op), pg, op) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f32[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(scvtf))]
pub fn svcvt_f32_s32_z(pg: svbool_t, op: svint32_t) -> svfloat32_t {
    svcvt_f32_s32_m(svdup_n_f32(0.0), pg, op)
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f32[_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(scvtf))]
pub fn svcvt_f32_s64_m(inactive: svfloat32_t, pg: svbool_t, op: svint64_t) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.scvtf.f32i64")]
        fn _svcvt_f32_s64_m(inactive: svfloat32_t, pg: svbool2_t, op: svint64_t) -> svfloat32_t;
    }
    unsafe { _svcvt_f32_s64_m(inactive, simd_cast(pg), op) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f32[_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(scvtf))]
pub fn svcvt_f32_s64_x(pg: svbool_t, op: svint64_t) -> svfloat32_t {
    unsafe { svcvt_f32_s64_m(simd_reinterpret(op), pg, op) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f32[_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(scvtf))]
pub fn svcvt_f32_s64_z(pg: svbool_t, op: svint64_t) -> svfloat32_t {
    svcvt_f32_s64_m(svdup_n_f32(0.0), pg, op)
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f32[_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub fn svcvt_f32_u32_m(inactive: svfloat32_t, pg: svbool_t, op: svuint32_t) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.ucvtf.nxv4f32.nxv4i32"
        )]
        fn _svcvt_f32_u32_m(inactive: svfloat32_t, pg: svbool4_t, op: svint32_t) -> svfloat32_t;
    }
    unsafe { _svcvt_f32_u32_m(inactive, simd_cast(pg), op.as_signed()) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f32[_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub fn svcvt_f32_u32_x(pg: svbool_t, op: svuint32_t) -> svfloat32_t {
    unsafe { svcvt_f32_u32_m(simd_reinterpret(op), pg, op) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f32[_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub fn svcvt_f32_u32_z(pg: svbool_t, op: svuint32_t) -> svfloat32_t {
    svcvt_f32_u32_m(svdup_n_f32(0.0), pg, op)
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f32[_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub fn svcvt_f32_u64_m(inactive: svfloat32_t, pg: svbool_t, op: svuint64_t) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ucvtf.f32i64")]
        fn _svcvt_f32_u64_m(inactive: svfloat32_t, pg: svbool2_t, op: svint64_t) -> svfloat32_t;
    }
    unsafe { _svcvt_f32_u64_m(inactive, simd_cast(pg), op.as_signed()) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f32[_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub fn svcvt_f32_u64_x(pg: svbool_t, op: svuint64_t) -> svfloat32_t {
    unsafe { svcvt_f32_u64_m(simd_reinterpret(op), pg, op) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f32[_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub fn svcvt_f32_u64_z(pg: svbool_t, op: svuint64_t) -> svfloat32_t {
    svcvt_f32_u64_m(svdup_n_f32(0.0), pg, op)
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f64[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(scvtf))]
pub fn svcvt_f64_s32_m(inactive: svfloat64_t, pg: svbool_t, op: svint32_t) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.scvtf.nxv2f64.nxv4i32"
        )]
        fn _svcvt_f64_s32_m(inactive: svfloat64_t, pg: svbool2_t, op: svint32_t) -> svfloat64_t;
    }
    unsafe { _svcvt_f64_s32_m(inactive, simd_cast(pg), op) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f64[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(scvtf))]
pub fn svcvt_f64_s32_x(pg: svbool_t, op: svint32_t) -> svfloat64_t {
    unsafe { svcvt_f64_s32_m(simd_reinterpret(op), pg, op) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f64[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(scvtf))]
pub fn svcvt_f64_s32_z(pg: svbool_t, op: svint32_t) -> svfloat64_t {
    svcvt_f64_s32_m(svdup_n_f64(0.0), pg, op)
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f64[_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(scvtf))]
pub fn svcvt_f64_s64_m(inactive: svfloat64_t, pg: svbool_t, op: svint64_t) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.scvtf.nxv2f64.nxv2i64"
        )]
        fn _svcvt_f64_s64_m(inactive: svfloat64_t, pg: svbool2_t, op: svint64_t) -> svfloat64_t;
    }
    unsafe { _svcvt_f64_s64_m(inactive, simd_cast(pg), op) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f64[_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(scvtf))]
pub fn svcvt_f64_s64_x(pg: svbool_t, op: svint64_t) -> svfloat64_t {
    unsafe { svcvt_f64_s64_m(simd_reinterpret(op), pg, op) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f64[_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(scvtf))]
pub fn svcvt_f64_s64_z(pg: svbool_t, op: svint64_t) -> svfloat64_t {
    svcvt_f64_s64_m(svdup_n_f64(0.0), pg, op)
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f64[_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub fn svcvt_f64_u32_m(inactive: svfloat64_t, pg: svbool_t, op: svuint32_t) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.ucvtf.nxv2f64.nxv4i32"
        )]
        fn _svcvt_f64_u32_m(inactive: svfloat64_t, pg: svbool2_t, op: svint32_t) -> svfloat64_t;
    }
    unsafe { _svcvt_f64_u32_m(inactive, simd_cast(pg), op.as_signed()) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f64[_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub fn svcvt_f64_u32_x(pg: svbool_t, op: svuint32_t) -> svfloat64_t {
    unsafe { svcvt_f64_u32_m(simd_reinterpret(op), pg, op) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f64[_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub fn svcvt_f64_u32_z(pg: svbool_t, op: svuint32_t) -> svfloat64_t {
    svcvt_f64_u32_m(svdup_n_f64(0.0), pg, op)
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f64[_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub fn svcvt_f64_u64_m(inactive: svfloat64_t, pg: svbool_t, op: svuint64_t) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.ucvtf.nxv2f64.nxv2i64"
        )]
        fn _svcvt_f64_u64_m(inactive: svfloat64_t, pg: svbool2_t, op: svint64_t) -> svfloat64_t;
    }
    unsafe { _svcvt_f64_u64_m(inactive, simd_cast(pg), op.as_signed()) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f64[_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub fn svcvt_f64_u64_x(pg: svbool_t, op: svuint64_t) -> svfloat64_t {
    unsafe { svcvt_f64_u64_m(simd_reinterpret(op), pg, op) }
}
#[doc = "Floating-point convert"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcvt_f64[_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ucvtf))]
pub fn svcvt_f64_u64_z(pg: svbool_t, op: svuint64_t) -> svfloat64_t {
    svcvt_f64_u64_m(svdup_n_f64(0.0), pg, op)
}
#[doc = "Broadcast a scalar value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdup[_n]_f32)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mov))]
pub fn svdup_n_f32(op: f32) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv4f32")]
        fn _svdup_n_f32(op: f32) -> svfloat32_t;
    }
    unsafe { _svdup_n_f32(op) }
}
#[doc = "Broadcast a scalar value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdup[_n]_f64)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mov))]
pub fn svdup_n_f64(op: f64) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv2f64")]
        fn _svdup_n_f64(op: f64) -> svfloat64_t;
    }
    unsafe { _svdup_n_f64(op) }
}
#[doc = "Broadcast a scalar value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdup[_n]_s8)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mov))]
pub fn svdup_n_s8(op: i8) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv16i8")]
        fn _svdup_n_s8(op: i8) -> svint8_t;
    }
    unsafe { _svdup_n_s8(op) }
}
#[doc = "Broadcast a scalar value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdup[_n]_s16)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mov))]
pub fn svdup_n_s16(op: i16) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv8i16")]
        fn _svdup_n_s16(op: i16) -> svint16_t;
    }
    unsafe { _svdup_n_s16(op) }
}
#[doc = "Broadcast a scalar value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdup[_n]_s32)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mov))]
pub fn svdup_n_s32(op: i32) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv4i32")]
        fn _svdup_n_s32(op: i32) -> svint32_t;
    }
    unsafe { _svdup_n_s32(op) }
}
#[doc = "Broadcast a scalar value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdup[_n]_s64)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mov))]
pub fn svdup_n_s64(op: i64) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv2i64")]
        fn _svdup_n_s64(op: i64) -> svint64_t;
    }
    unsafe { _svdup_n_s64(op) }
}
#[doc = "Broadcast a scalar value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdup[_n]_u8)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mov))]
pub fn svdup_n_u8(op: u8) -> svuint8_t {
    unsafe { svdup_n_s8(op.as_signed()).as_unsigned() }
}
#[doc = "Broadcast a scalar value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdup[_n]_u16)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mov))]
pub fn svdup_n_u16(op: u16) -> svuint16_t {
    unsafe { svdup_n_s16(op.as_signed()).as_unsigned() }
}
#[doc = "Broadcast a scalar value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdup[_n]_u32)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mov))]
pub fn svdup_n_u32(op: u32) -> svuint32_t {
    unsafe { svdup_n_s32(op.as_signed()).as_unsigned() }
}
#[doc = "Broadcast a scalar value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdup[_n]_u64)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mov))]
pub fn svdup_n_u64(op: u64) -> svuint64_t {
    unsafe { svdup_n_s64(op.as_signed()).as_unsigned() }
}
#[doc = "Unextended load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1[_f32])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1w))]
pub unsafe fn svld1_f32(pg: svbool_t, base: *const f32) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.nxv4f32")]
        fn _svld1_f32(pg: svbool4_t, base: *const f32) -> svfloat32_t;
    }
    _svld1_f32(simd_cast(pg), base)
}
#[doc = "Unextended load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1[_f64])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1d))]
pub unsafe fn svld1_f64(pg: svbool_t, base: *const f64) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.nxv2f64")]
        fn _svld1_f64(pg: svbool2_t, base: *const f64) -> svfloat64_t;
    }
    _svld1_f64(simd_cast(pg), base)
}
#[doc = "Unextended load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1[_s8])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1b))]
pub unsafe fn svld1_s8(pg: svbool_t, base: *const i8) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.nxv16i8")]
        fn _svld1_s8(pg: svbool_t, base: *const i8) -> svint8_t;
    }
    _svld1_s8(pg, base)
}
#[doc = "Unextended load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1[_s16])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1h))]
pub unsafe fn svld1_s16(pg: svbool_t, base: *const i16) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.nxv8i16")]
        fn _svld1_s16(pg: svbool8_t, base: *const i16) -> svint16_t;
    }
    _svld1_s16(simd_cast(pg), base)
}
#[doc = "Unextended load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1[_s32])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1w))]
pub unsafe fn svld1_s32(pg: svbool_t, base: *const i32) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.nxv4i32")]
        fn _svld1_s32(pg: svbool4_t, base: *const i32) -> svint32_t;
    }
    _svld1_s32(simd_cast(pg), base)
}
#[doc = "Unextended load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1[_s64])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1d))]
pub unsafe fn svld1_s64(pg: svbool_t, base: *const i64) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.nxv2i64")]
        fn _svld1_s64(pg: svbool2_t, base: *const i64) -> svint64_t;
    }
    _svld1_s64(simd_cast(pg), base)
}
#[doc = "Unextended load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1[_u8])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1b))]
pub unsafe fn svld1_u8(pg: svbool_t, base: *const u8) -> svuint8_t {
    svld1_s8(pg, base.as_signed()).as_unsigned()
}
#[doc = "Unextended load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1[_u16])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1h))]
pub unsafe fn svld1_u16(pg: svbool_t, base: *const u16) -> svuint16_t {
    svld1_s16(pg, base.as_signed()).as_unsigned()
}
#[doc = "Unextended load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1[_u32])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1w))]
pub unsafe fn svld1_u32(pg: svbool_t, base: *const u32) -> svuint32_t {
    svld1_s32(pg, base.as_signed()).as_unsigned()
}
#[doc = "Unextended load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1[_u64])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1d))]
pub unsafe fn svld1_u64(pg: svbool_t, base: *const u64) -> svuint64_t {
    svld1_s64(pg, base.as_signed()).as_unsigned()
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fmul))]
pub fn svmul_f32_m(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fmul.nxv4f32")]
        fn _svmul_f32_m(pg: svbool4_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t;
    }
    unsafe { _svmul_f32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fmul))]
pub fn svmul_n_f32_m(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svmul_f32_m(pg, op1, svdup_n_f32(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fmul))]
pub fn svmul_f32_x(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    svmul_f32_m(pg, op1, op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fmul))]
pub fn svmul_n_f32_x(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svmul_f32_x(pg, op1, svdup_n_f32(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fmul))]
pub fn svmul_f32_z(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    svmul_f32_m(pg, svsel_f32(pg, op1, svdup_n_f32(0.0)), op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fmul))]
pub fn svmul_n_f32_z(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svmul_f32_z(pg, op1, svdup_n_f32(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fmul))]
pub fn svmul_f64_m(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fmul.nxv2f64")]
        fn _svmul_f64_m(pg: svbool2_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t;
    }
    unsafe { _svmul_f64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fmul))]
pub fn svmul_n_f64_m(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svmul_f64_m(pg, op1, svdup_n_f64(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fmul))]
pub fn svmul_f64_x(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    svmul_f64_m(pg, op1, op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fmul))]
pub fn svmul_n_f64_x(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svmul_f64_x(pg, op1, svdup_n_f64(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fmul))]
pub fn svmul_f64_z(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    svmul_f64_m(pg, svsel_f64(pg, op1, svdup_n_f64(0.0)), op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fmul))]
pub fn svmul_n_f64_z(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svmul_f64_z(pg, op1, svdup_n_f64(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.mul.nxv16i8")]
        fn _svmul_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t;
    }
    unsafe { _svmul_s8_m(pg, op1, op2) }
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_s8_m(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svmul_s8_m(pg, op1, svdup_n_s8(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_s8_x(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svmul_s8_m(pg, op1, op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_s8_x(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svmul_s8_x(pg, op1, svdup_n_s8(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_s8_z(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svmul_s8_m(pg, svsel_s8(pg, op1, svdup_n_s8(0)), op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_s8_z(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svmul_s8_z(pg, op1, svdup_n_s8(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_s16_m(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.mul.nxv8i16")]
        fn _svmul_s16_m(pg: svbool8_t, op1: svint16_t, op2: svint16_t) -> svint16_t;
    }
    unsafe { _svmul_s16_m(simd_cast(pg), op1, op2) }
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_s16_m(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svmul_s16_m(pg, op1, svdup_n_s16(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_s16_x(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svmul_s16_m(pg, op1, op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_s16_x(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svmul_s16_x(pg, op1, svdup_n_s16(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_s16_z(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svmul_s16_m(pg, svsel_s16(pg, op1, svdup_n_s16(0)), op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_s16_z(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svmul_s16_z(pg, op1, svdup_n_s16(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_s32_m(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.mul.nxv4i32")]
        fn _svmul_s32_m(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svint32_t;
    }
    unsafe { _svmul_s32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_s32_m(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svmul_s32_m(pg, op1, svdup_n_s32(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_s32_x(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svmul_s32_m(pg, op1, op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_s32_x(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svmul_s32_x(pg, op1, svdup_n_s32(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_s32_z(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svmul_s32_m(pg, svsel_s32(pg, op1, svdup_n_s32(0)), op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_s32_z(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svmul_s32_z(pg, op1, svdup_n_s32(op2))
}
#[doc = "Divide"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdiv[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sdiv))]
pub fn svdiv_s32_m(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sdiv.nxv4i32")]
        fn _svdiv_s32_m(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svint32_t;
    }
    unsafe { _svdiv_s32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Divide"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdiv[_n_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sdiv))]
pub fn svdiv_n_s32_m(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svdiv_s32_m(pg, op1, svdup_n_s32(op2))
}
#[doc = "Divide"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdiv[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sdiv))]
pub fn svdiv_s32_x(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svdiv_s32_m(pg, op1, op2)
}
#[doc = "Divide"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdiv[_n_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sdiv))]
pub fn svdiv_n_s32_x(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svdiv_s32_x(pg, op1, svdup_n_s32(op2))
}
#[doc = "Divide"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdiv[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sdiv))]
pub fn svdiv_s32_z(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svdiv_s32_m(pg, svsel_s32(pg, op1, svdup_n_s32(0)), op2)
}
#[doc = "Divide"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svdiv[_n_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sdiv))]
pub fn svdiv_n_s32_z(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svdiv_s32_z(pg, op1, svdup_n_s32(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_s64_m(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.mul.nxv2i64")]
        fn _svmul_s64_m(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svint64_t;
    }
    unsafe { _svmul_s64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_s64_m(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svmul_s64_m(pg, op1, svdup_n_s64(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_s64_x(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svmul_s64_m(pg, op1, op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_s64_x(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svmul_s64_x(pg, op1, svdup_n_s64(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_s64_z(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svmul_s64_m(pg, svsel_s64(pg, op1, svdup_n_s64(0)), op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_s64_z(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svmul_s64_z(pg, op1, svdup_n_s64(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_u8_m(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    unsafe { svmul_s8_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_u8_m(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svmul_u8_m(pg, op1, svdup_n_u8(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_u8_x(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svmul_u8_m(pg, op1, op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_u8_x(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svmul_u8_x(pg, op1, svdup_n_u8(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_u8_z(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svmul_u8_m(pg, svsel_u8(pg, op1, svdup_n_u8(0)), op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_u8_z(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svmul_u8_z(pg, op1, svdup_n_u8(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_u16_m(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    unsafe { svmul_s16_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_u16_m(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svmul_u16_m(pg, op1, svdup_n_u16(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_u16_x(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svmul_u16_m(pg, op1, op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_u16_x(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svmul_u16_x(pg, op1, svdup_n_u16(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_u16_z(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svmul_u16_m(pg, svsel_u16(pg, op1, svdup_n_u16(0)), op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_u16_z(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svmul_u16_z(pg, op1, svdup_n_u16(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_u32_m(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    unsafe { svmul_s32_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_u32_m(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svmul_u32_m(pg, op1, svdup_n_u32(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_u32_x(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svmul_u32_m(pg, op1, op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_u32_x(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svmul_u32_x(pg, op1, svdup_n_u32(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_u32_z(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svmul_u32_m(pg, svsel_u32(pg, op1, svdup_n_u32(0)), op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_u32_z(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svmul_u32_z(pg, op1, svdup_n_u32(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_u64_m(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    unsafe { svmul_s64_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_u64_m(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svmul_u64_m(pg, op1, svdup_n_u64(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_u64_x(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svmul_u64_m(pg, op1, op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_u64_x(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svmul_u64_x(pg, op1, svdup_n_u64(op2))
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_u64_z(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svmul_u64_m(pg, svsel_u64(pg, op1, svdup_n_u64(0)), op2)
}
#[doc = "Multiply"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svmul[_n_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(mul))]
pub fn svmul_n_u64_z(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svmul_u64_z(pg, op1, svdup_n_u64(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.orr.nxv16i8")]
        fn _svorr_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t;
    }
    unsafe { _svorr_s8_m(pg, op1, op2) }
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_s8_m(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svorr_s8_m(pg, op1, svdup_n_s8(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_s8_x(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svorr_s8_m(pg, op1, op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_s8_x(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svorr_s8_x(pg, op1, svdup_n_s8(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_s8_z(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svorr_s8_m(pg, svsel_s8(pg, op1, svdup_n_s8(0)), op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_s8_z(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svorr_s8_z(pg, op1, svdup_n_s8(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_s16_m(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.orr.nxv8i16")]
        fn _svorr_s16_m(pg: svbool8_t, op1: svint16_t, op2: svint16_t) -> svint16_t;
    }
    unsafe { _svorr_s16_m(simd_cast(pg), op1, op2) }
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_s16_m(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svorr_s16_m(pg, op1, svdup_n_s16(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_s16_x(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svorr_s16_m(pg, op1, op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_s16_x(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svorr_s16_x(pg, op1, svdup_n_s16(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_s16_z(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svorr_s16_m(pg, svsel_s16(pg, op1, svdup_n_s16(0)), op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_s16_z(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svorr_s16_z(pg, op1, svdup_n_s16(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_s32_m(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.orr.nxv4i32")]
        fn _svorr_s32_m(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svint32_t;
    }
    unsafe { _svorr_s32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_s32_m(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svorr_s32_m(pg, op1, svdup_n_s32(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_s32_x(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svorr_s32_m(pg, op1, op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_s32_x(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svorr_s32_x(pg, op1, svdup_n_s32(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_s32_z(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svorr_s32_m(pg, svsel_s32(pg, op1, svdup_n_s32(0)), op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_s32_z(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svorr_s32_z(pg, op1, svdup_n_s32(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_s64_m(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.orr.nxv2i64")]
        fn _svorr_s64_m(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svint64_t;
    }
    unsafe { _svorr_s64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_s64_m(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svorr_s64_m(pg, op1, svdup_n_s64(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_s64_x(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svorr_s64_m(pg, op1, op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_s64_x(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svorr_s64_x(pg, op1, svdup_n_s64(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_s64_z(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svorr_s64_m(pg, svsel_s64(pg, op1, svdup_n_s64(0)), op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_s64_z(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svorr_s64_z(pg, op1, svdup_n_s64(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_u8_m(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    unsafe { svorr_s8_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_u8_m(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svorr_u8_m(pg, op1, svdup_n_u8(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_u8_x(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svorr_u8_m(pg, op1, op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_u8_x(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svorr_u8_x(pg, op1, svdup_n_u8(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_u8_z(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svorr_u8_m(pg, svsel_u8(pg, op1, svdup_n_u8(0)), op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_u8_z(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svorr_u8_z(pg, op1, svdup_n_u8(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_u16_m(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    unsafe { svorr_s16_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_u16_m(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svorr_u16_m(pg, op1, svdup_n_u16(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_u16_x(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svorr_u16_m(pg, op1, op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_u16_x(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svorr_u16_x(pg, op1, svdup_n_u16(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_u16_z(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svorr_u16_m(pg, svsel_u16(pg, op1, svdup_n_u16(0)), op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_u16_z(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svorr_u16_z(pg, op1, svdup_n_u16(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_u32_m(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    unsafe { svorr_s32_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_u32_m(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svorr_u32_m(pg, op1, svdup_n_u32(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_u32_x(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svorr_u32_m(pg, op1, op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_u32_x(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svorr_u32_x(pg, op1, svdup_n_u32(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_u32_z(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svorr_u32_m(pg, svsel_u32(pg, op1, svdup_n_u32(0)), op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_u32_z(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svorr_u32_z(pg, op1, svdup_n_u32(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_u64_m(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    unsafe { svorr_s64_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_u64_m(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svorr_u64_m(pg, op1, svdup_n_u64(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_u64_x(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svorr_u64_m(pg, op1, op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_u64_x(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svorr_u64_x(pg, op1, svdup_n_u64(op2))
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_u64_z(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svorr_u64_m(pg, svsel_u64(pg, op1, svdup_n_u64(0)), op2)
}
#[doc = "Bitwise inclusive OR"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svorr[_n_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(orr))]
pub fn svorr_n_u64_z(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svorr_u64_z(pg, op1, svdup_n_u64(op2))
}
#[doc = "Set predicate elements to true"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svptrue_pat_b8)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
# [cfg_attr (test , assert_instr (ptrue , PATTERN = { svpattern :: SV_ALL }))]
pub fn svptrue_pat_b8<const PATTERN: svpattern>() -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ptrue.nxv16i1")]
        fn _svptrue_pat_b8(pattern: svpattern) -> svbool_t;
    }
    unsafe { _svptrue_pat_b8(PATTERN) }
}
#[doc = "Set predicate elements to true"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svptrue_pat_b16)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
# [cfg_attr (test , assert_instr (ptrue , PATTERN = { svpattern :: SV_ALL }))]
pub fn svptrue_pat_b16<const PATTERN: svpattern>() -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ptrue.nxv8i1")]
        fn _svptrue_pat_b16(pattern: svpattern) -> svbool8_t;
    }
    unsafe { simd_cast(_svptrue_pat_b16(PATTERN)) }
}
#[doc = "Set predicate elements to true"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svptrue_pat_b32)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
# [cfg_attr (test , assert_instr (ptrue , PATTERN = { svpattern :: SV_ALL }))]
pub fn svptrue_pat_b32<const PATTERN: svpattern>() -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ptrue.nxv4i1")]
        fn _svptrue_pat_b32(pattern: svpattern) -> svbool4_t;
    }
    unsafe { simd_cast(_svptrue_pat_b32(PATTERN)) }
}
#[doc = "Set predicate elements to true"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svptrue_pat_b64)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
# [cfg_attr (test , assert_instr (ptrue , PATTERN = { svpattern :: SV_ALL }))]
pub fn svptrue_pat_b64<const PATTERN: svpattern>() -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ptrue.nxv2i1")]
        fn _svptrue_pat_b64(pattern: svpattern) -> svbool2_t;
    }
    unsafe { simd_cast(_svptrue_pat_b64(PATTERN)) }
}
#[doc = "Conditionally select elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsel[_b])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sel))]
pub fn svsel_b(pg: svbool_t, op1: svbool_t, op2: svbool_t) -> svbool_t {
    unsafe { simd_select(simd_cast::<_, svbool_t>(pg), op1, op2) }
}
#[doc = "Conditionally select elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsel[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sel))]
pub fn svsel_f32(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    unsafe { simd_select(simd_cast::<_, svbool4_t>(pg), op1, op2) }
}
#[doc = "Conditionally select elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsel[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sel))]
pub fn svsel_f64(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    unsafe { simd_select(simd_cast::<_, svbool2_t>(pg), op1, op2) }
}
#[doc = "Conditionally select elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsel[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sel))]
pub fn svsel_s8(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    unsafe { simd_select(simd_cast::<_, svbool_t>(pg), op1, op2) }
}
#[doc = "Conditionally select elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsel[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sel))]
pub fn svsel_s16(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    unsafe { simd_select(simd_cast::<_, svbool2_t>(pg), op1, op2) }
}
#[doc = "Conditionally select elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsel[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sel))]
pub fn svsel_s32(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    unsafe { simd_select(simd_cast::<_, svbool4_t>(pg), op1, op2) }
}
#[doc = "Conditionally select elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsel[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sel))]
pub fn svsel_s64(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    unsafe { simd_select(simd_cast::<_, svbool8_t>(pg), op1, op2) }
}
#[doc = "Conditionally select elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsel[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sel))]
pub fn svsel_u8(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    unsafe { simd_select(simd_cast::<_, svbool_t>(pg), op1, op2) }
}
#[doc = "Conditionally select elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsel[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sel))]
pub fn svsel_u16(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    unsafe { simd_select(simd_cast::<_, svbool2_t>(pg), op1, op2) }
}
#[doc = "Conditionally select elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsel[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sel))]
pub fn svsel_u32(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    unsafe { simd_select(simd_cast::<_, svbool4_t>(pg), op1, op2) }
}
#[doc = "Conditionally select elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsel[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sel))]
pub fn svsel_u64(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    unsafe { simd_select(simd_cast::<_, svbool8_t>(pg), op1, op2) }
}
#[doc = "Non-truncating store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1[_f32])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1w))]
pub unsafe fn svst1_f32(pg: svbool_t, base: *mut f32, data: svfloat32_t) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.nxv4f32")]
        fn _svst1_f32(data: svfloat32_t, pg: svbool4_t, ptr: *mut f32);
    }
    _svst1_f32(data, simd_cast(pg), base)
}
#[doc = "Non-truncating store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1[_f64])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1d))]
pub unsafe fn svst1_f64(pg: svbool_t, base: *mut f64, data: svfloat64_t) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.nxv2f64")]
        fn _svst1_f64(data: svfloat64_t, pg: svbool2_t, ptr: *mut f64);
    }
    _svst1_f64(data, simd_cast(pg), base)
}
#[doc = "Non-truncating store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1[_s8])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1b))]
pub unsafe fn svst1_s8(pg: svbool_t, base: *mut i8, data: svint8_t) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.nxv16i8")]
        fn _svst1_s8(data: svint8_t, pg: svbool_t, ptr: *mut i8);
    }
    _svst1_s8(data, pg, base)
}
#[doc = "Non-truncating store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1[_s16])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1h))]
pub unsafe fn svst1_s16(pg: svbool_t, base: *mut i16, data: svint16_t) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.nxv8i16")]
        fn _svst1_s16(data: svint16_t, pg: svbool8_t, ptr: *mut i16);
    }
    _svst1_s16(data, simd_cast(pg), base)
}
#[doc = "Non-truncating store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1[_s32])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1w))]
pub unsafe fn svst1_s32(pg: svbool_t, base: *mut i32, data: svint32_t) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.nxv4i32")]
        fn _svst1_s32(data: svint32_t, pg: svbool4_t, ptr: *mut i32);
    }
    _svst1_s32(data, simd_cast(pg), base)
}
#[doc = "Non-truncating store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1[_s64])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1d))]
pub unsafe fn svst1_s64(pg: svbool_t, base: *mut i64, data: svint64_t) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.nxv2i64")]
        fn _svst1_s64(data: svint64_t, pg: svbool2_t, ptr: *mut i64);
    }
    _svst1_s64(data, simd_cast(pg), base)
}
#[doc = "Non-truncating store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1[_u8])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1b))]
pub unsafe fn svst1_u8(pg: svbool_t, base: *mut u8, data: svuint8_t) {
    svst1_s8(pg, base.as_signed(), data.as_signed())
}
#[doc = "Non-truncating store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1[_u16])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1h))]
pub unsafe fn svst1_u16(pg: svbool_t, base: *mut u16, data: svuint16_t) {
    svst1_s16(pg, base.as_signed(), data.as_signed())
}
#[doc = "Non-truncating store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1[_u32])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1w))]
pub unsafe fn svst1_u32(pg: svbool_t, base: *mut u32, data: svuint32_t) {
    svst1_s32(pg, base.as_signed(), data.as_signed())
}
#[doc = "Non-truncating store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1[_u64])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1d))]
pub unsafe fn svst1_u64(pg: svbool_t, base: *mut u64, data: svuint64_t) {
    svst1_s64(pg, base.as_signed(), data.as_signed())
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsub))]
pub fn svsub_f32_m(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fsub.nxv4f32")]
        fn _svsub_f32_m(pg: svbool4_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t;
    }
    unsafe { _svsub_f32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsub))]
pub fn svsub_n_f32_m(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svsub_f32_m(pg, op1, svdup_n_f32(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsub))]
pub fn svsub_f32_x(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    svsub_f32_m(pg, op1, op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsub))]
pub fn svsub_n_f32_x(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svsub_f32_x(pg, op1, svdup_n_f32(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsub))]
pub fn svsub_f32_z(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    svsub_f32_m(pg, svsel_f32(pg, op1, svdup_n_f32(0.0)), op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsub))]
pub fn svsub_n_f32_z(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svsub_f32_z(pg, op1, svdup_n_f32(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsub))]
pub fn svsub_f64_m(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fsub.nxv2f64")]
        fn _svsub_f64_m(pg: svbool2_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t;
    }
    unsafe { _svsub_f64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsub))]
pub fn svsub_n_f64_m(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svsub_f64_m(pg, op1, svdup_n_f64(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsub))]
pub fn svsub_f64_x(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    svsub_f64_m(pg, op1, op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsub))]
pub fn svsub_n_f64_x(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svsub_f64_x(pg, op1, svdup_n_f64(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsub))]
pub fn svsub_f64_z(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    svsub_f64_m(pg, svsel_f64(pg, op1, svdup_n_f64(0.0)), op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsub))]
pub fn svsub_n_f64_z(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svsub_f64_z(pg, op1, svdup_n_f64(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sub.nxv16i8")]
        fn _svsub_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t;
    }
    unsafe { _svsub_s8_m(pg, op1, op2) }
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_s8_m(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svsub_s8_m(pg, op1, svdup_n_s8(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_s8_x(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svsub_s8_m(pg, op1, op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_s8_x(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svsub_s8_x(pg, op1, svdup_n_s8(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_s8_z(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svsub_s8_m(pg, svsel_s8(pg, op1, svdup_n_s8(0)), op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_s8_z(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svsub_s8_z(pg, op1, svdup_n_s8(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_s16_m(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sub.nxv8i16")]
        fn _svsub_s16_m(pg: svbool8_t, op1: svint16_t, op2: svint16_t) -> svint16_t;
    }
    unsafe { _svsub_s16_m(simd_cast(pg), op1, op2) }
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_s16_m(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svsub_s16_m(pg, op1, svdup_n_s16(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_s16_x(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svsub_s16_m(pg, op1, op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_s16_x(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svsub_s16_x(pg, op1, svdup_n_s16(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_s16_z(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svsub_s16_m(pg, svsel_s16(pg, op1, svdup_n_s16(0)), op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_s16_z(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svsub_s16_z(pg, op1, svdup_n_s16(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_s32_m(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sub.nxv4i32")]
        fn _svsub_s32_m(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svint32_t;
    }
    unsafe { _svsub_s32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_s32_m(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svsub_s32_m(pg, op1, svdup_n_s32(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_s32_x(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svsub_s32_m(pg, op1, op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_s32_x(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svsub_s32_x(pg, op1, svdup_n_s32(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_s32_z(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svsub_s32_m(pg, svsel_s32(pg, op1, svdup_n_s32(0)), op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_s32_z(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svsub_s32_z(pg, op1, svdup_n_s32(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_s64_m(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sub.nxv2i64")]
        fn _svsub_s64_m(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svint64_t;
    }
    unsafe { _svsub_s64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_s64_m(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svsub_s64_m(pg, op1, svdup_n_s64(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_s64_x(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svsub_s64_m(pg, op1, op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_s64_x(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svsub_s64_x(pg, op1, svdup_n_s64(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_s64_z(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svsub_s64_m(pg, svsel_s64(pg, op1, svdup_n_s64(0)), op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_s64_z(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svsub_s64_z(pg, op1, svdup_n_s64(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_u8_m(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    unsafe { svsub_s8_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_u8_m(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svsub_u8_m(pg, op1, svdup_n_u8(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_u8_x(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svsub_u8_m(pg, op1, op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_u8_x(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svsub_u8_x(pg, op1, svdup_n_u8(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_u8_z(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svsub_u8_m(pg, svsel_u8(pg, op1, svdup_n_u8(0)), op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_u8_z(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svsub_u8_z(pg, op1, svdup_n_u8(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_u16_m(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    unsafe { svsub_s16_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_u16_m(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svsub_u16_m(pg, op1, svdup_n_u16(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_u16_x(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svsub_u16_m(pg, op1, op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_u16_x(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svsub_u16_x(pg, op1, svdup_n_u16(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_u16_z(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svsub_u16_m(pg, svsel_u16(pg, op1, svdup_n_u16(0)), op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_u16_z(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svsub_u16_z(pg, op1, svdup_n_u16(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_u32_m(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    unsafe { svsub_s32_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_u32_m(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svsub_u32_m(pg, op1, svdup_n_u32(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_u32_x(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svsub_u32_m(pg, op1, op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_u32_x(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svsub_u32_x(pg, op1, svdup_n_u32(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_u32_z(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svsub_u32_m(pg, svsel_u32(pg, op1, svdup_n_u32(0)), op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_u32_z(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svsub_u32_z(pg, op1, svdup_n_u32(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_u64_m(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    unsafe { svsub_s64_m(pg, op1.as_signed(), op2.as_signed()).as_unsigned() }
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_u64_m(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svsub_u64_m(pg, op1, svdup_n_u64(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_u64_x(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svsub_u64_m(pg, op1, op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_u64_x(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svsub_u64_x(pg, op1, svdup_n_u64(op2))
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_u64_z(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svsub_u64_m(pg, svsel_u64(pg, op1, svdup_n_u64(0)), op2)
}
#[doc = "Subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsub[_n_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sub))]
pub fn svsub_n_u64_z(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svsub_u64_z(pg, op1, svdup_n_u64(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabd))]
pub fn svabd_f32_m(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fabd.nxv4f32")]
        fn _svabd_f32_m(pg: svbool4_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t;
    }
    unsafe { _svabd_f32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabd))]
pub fn svabd_n_f32_m(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svabd_f32_m(pg, op1, svdup_n_f32(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabd))]
pub fn svabd_f32_x(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    svabd_f32_m(pg, op1, op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabd))]
pub fn svabd_n_f32_x(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svabd_f32_x(pg, op1, svdup_n_f32(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabd))]
pub fn svabd_f32_z(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    svabd_f32_m(pg, svsel_f32(pg, op1, svdup_n_f32(0.0)), op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabd))]
pub fn svabd_n_f32_z(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svabd_f32_z(pg, op1, svdup_n_f32(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabd))]
pub fn svabd_f64_m(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fabd.nxv2f64")]
        fn _svabd_f64_m(pg: svbool2_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t;
    }
    unsafe { _svabd_f64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabd))]
pub fn svabd_n_f64_m(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svabd_f64_m(pg, op1, svdup_n_f64(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabd))]
pub fn svabd_f64_x(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    svabd_f64_m(pg, op1, op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabd))]
pub fn svabd_n_f64_x(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svabd_f64_x(pg, op1, svdup_n_f64(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabd))]
pub fn svabd_f64_z(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    svabd_f64_m(pg, svsel_f64(pg, op1, svdup_n_f64(0.0)), op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabd))]
pub fn svabd_n_f64_z(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svabd_f64_z(pg, op1, svdup_n_f64(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sabd.nxv16i8")]
        fn _svabd_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t;
    }
    unsafe { _svabd_s8_m(pg, op1, op2) }
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_n_s8_m(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svabd_s8_m(pg, op1, svdup_n_s8(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_s8_x(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svabd_s8_m(pg, op1, op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_n_s8_x(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svabd_s8_x(pg, op1, svdup_n_s8(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_s8_z(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svabd_s8_m(pg, svsel_s8(pg, op1, svdup_n_s8(0)), op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_n_s8_z(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svabd_s8_z(pg, op1, svdup_n_s8(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_s16_m(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sabd.nxv8i16")]
        fn _svabd_s16_m(pg: svbool8_t, op1: svint16_t, op2: svint16_t) -> svint16_t;
    }
    unsafe { _svabd_s16_m(simd_cast(pg), op1, op2) }
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_n_s16_m(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svabd_s16_m(pg, op1, svdup_n_s16(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_s16_x(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svabd_s16_m(pg, op1, op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_n_s16_x(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svabd_s16_x(pg, op1, svdup_n_s16(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_s16_z(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svabd_s16_m(pg, svsel_s16(pg, op1, svdup_n_s16(0)), op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_n_s16_z(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svabd_s16_z(pg, op1, svdup_n_s16(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_s32_m(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sabd.nxv4i32")]
        fn _svabd_s32_m(pg: svbool4_t, op1: svint32_t, op2: svint32_t) -> svint32_t;
    }
    unsafe { _svabd_s32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_n_s32_m(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svabd_s32_m(pg, op1, svdup_n_s32(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_s32_x(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svabd_s32_m(pg, op1, op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_n_s32_x(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svabd_s32_x(pg, op1, svdup_n_s32(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_s32_z(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svabd_s32_m(pg, svsel_s32(pg, op1, svdup_n_s32(0)), op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_n_s32_z(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svabd_s32_z(pg, op1, svdup_n_s32(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_s64_m(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sabd.nxv2i64")]
        fn _svabd_s64_m(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svint64_t;
    }
    unsafe { _svabd_s64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_n_s64_m(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svabd_s64_m(pg, op1, svdup_n_s64(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_s64_x(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svabd_s64_m(pg, op1, op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_n_s64_x(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svabd_s64_x(pg, op1, svdup_n_s64(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_s64_z(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svabd_s64_m(pg, svsel_s64(pg, op1, svdup_n_s64(0)), op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sabd))]
pub fn svabd_n_s64_z(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svabd_s64_z(pg, op1, svdup_n_s64(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_u8_m(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.uabd.nxv16i8")]
        fn _svabd_u8_m(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t;
    }
    unsafe { _svabd_u8_m(pg, op1, op2) }
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_n_u8_m(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svabd_u8_m(pg, op1, svdup_n_u8(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_u8_x(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svabd_u8_m(pg, op1, op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_n_u8_x(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svabd_u8_x(pg, op1, svdup_n_u8(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_u8_z(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svabd_u8_m(pg, svsel_u8(pg, op1, svdup_n_u8(0)), op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_n_u8_z(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svabd_u8_z(pg, op1, svdup_n_u8(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_u16_m(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.uabd.nxv8i16")]
        fn _svabd_u16_m(pg: svbool8_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t;
    }
    unsafe { _svabd_u16_m(simd_cast(pg), op1, op2) }
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_n_u16_m(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svabd_u16_m(pg, op1, svdup_n_u16(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_u16_x(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svabd_u16_m(pg, op1, op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_n_u16_x(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svabd_u16_x(pg, op1, svdup_n_u16(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_u16_z(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svabd_u16_m(pg, svsel_u16(pg, op1, svdup_n_u16(0)), op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_n_u16_z(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svabd_u16_z(pg, op1, svdup_n_u16(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_u32_m(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.uabd.nxv4i32")]
        fn _svabd_u32_m(pg: svbool4_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t;
    }
    unsafe { _svabd_u32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_n_u32_m(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svabd_u32_m(pg, op1, svdup_n_u32(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_u32_x(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svabd_u32_m(pg, op1, op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_n_u32_x(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svabd_u32_x(pg, op1, svdup_n_u32(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_u32_z(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svabd_u32_m(pg, svsel_u32(pg, op1, svdup_n_u32(0)), op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_n_u32_z(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svabd_u32_z(pg, op1, svdup_n_u32(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_u64_m(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.uabd.nxv2i64")]
        fn _svabd_u64_m(pg: svbool2_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t;
    }
    unsafe { _svabd_u64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_n_u64_m(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svabd_u64_m(pg, op1, svdup_n_u64(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_u64_x(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svabd_u64_m(pg, op1, op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_n_u64_x(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svabd_u64_x(pg, op1, svdup_n_u64(op2))
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_u64_z(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svabd_u64_m(pg, svsel_u64(pg, op1, svdup_n_u64(0)), op2)
}
#[doc = "Absolute difference"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabd[_n_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uabd))]
pub fn svabd_n_u64_z(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svabd_u64_z(pg, op1, svdup_n_u64(op2))
}

#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabs))]
pub fn svabs_f32_m(pg: svbool_t, op: svfloat32_t) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fabs.nxv4f32")]
        fn _svabs_f32_m(pg: svbool4_t, op: svfloat32_t) -> svfloat32_t;
    }
    unsafe { _svabs_f32_m(simd_cast(pg), op) }
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabs))]
pub fn svabs_f32_x(pg: svbool_t, op: svfloat32_t) -> svfloat32_t {
    svabs_f32_m(pg, op)
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabs))]
pub fn svabs_f32_z(pg: svbool_t, op: svfloat32_t) -> svfloat32_t {
    svabs_f32_m(pg, svsel_f32(pg, op, svdup_n_f32(0.0)))
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabs))]
pub fn svabs_f64_m(pg: svbool_t, op: svfloat64_t) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fabs.nxv2f64")]
        fn _svabs_f64_m(pg: svbool2_t, op: svfloat64_t) -> svfloat64_t;
    }
    unsafe { _svabs_f64_m(simd_cast(pg), op) }
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabs))]
pub fn svabs_f64_x(pg: svbool_t, op: svfloat64_t) -> svfloat64_t {
    svabs_f64_m(pg, op)
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fabs))]
pub fn svabs_f64_z(pg: svbool_t, op: svfloat64_t) -> svfloat64_t {
    svabs_f64_m(pg, svsel_f64(pg, op, svdup_n_f64(0.0)))
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(abs))]
pub fn svabs_s8_m(pg: svbool_t, op: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.abs.nxv16i8")]
        fn _svabs_s8_m(pg: svbool_t, op: svint8_t) -> svint8_t;
    }
    unsafe { _svabs_s8_m(pg, op) }
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(abs))]
pub fn svabs_s8_x(pg: svbool_t, op: svint8_t) -> svint8_t {
    svabs_s8_m(pg, op)
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(abs))]
pub fn svabs_s8_z(pg: svbool_t, op: svint8_t) -> svint8_t {
    svabs_s8_m(pg, svsel_s8(pg, op, svdup_n_s8(0)))
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(abs))]
pub fn svabs_s16_m(pg: svbool_t, op: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.abs.nxv8i16")]
        fn _svabs_s16_m(pg: svbool8_t, op: svint16_t) -> svint16_t;
    }
    unsafe { _svabs_s16_m(simd_cast(pg), op) }
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(abs))]
pub fn svabs_s16_x(pg: svbool_t, op: svint16_t) -> svint16_t {
    svabs_s16_m(pg, op)
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(abs))]
pub fn svabs_s16_z(pg: svbool_t, op: svint16_t) -> svint16_t {
    svabs_s16_m(pg, svsel_s16(pg, op, svdup_n_s16(0)))
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(abs))]
pub fn svabs_s32_m(pg: svbool_t, op: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.abs.nxv4i32")]
        fn _svabs_s32_m(pg: svbool4_t, op: svint32_t) -> svint32_t;
    }
    unsafe { _svabs_s32_m(simd_cast(pg), op) }
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(abs))]
pub fn svabs_s32_x(pg: svbool_t, op: svint32_t) -> svint32_t {
    svabs_s32_m(pg, op)
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(abs))]
pub fn svabs_s32_z(pg: svbool_t, op: svint32_t) -> svint32_t {
    svabs_s32_m(pg, svsel_s32(pg, op, svdup_n_s32(0)))
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(abs))]
pub fn svabs_s64_m(pg: svbool_t, op: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.abs.nxv2i64")]
        fn _svabs_s64_m(pg: svbool2_t, op: svint64_t) -> svint64_t;
    }
    unsafe { _svabs_s64_m(simd_cast(pg), op) }
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(abs))]
pub fn svabs_s64_x(pg: svbool_t, op: svint64_t) -> svint64_t {
    svabs_s64_m(pg, op)
}
#[doc = "Absolute value"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svabs[_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(abs))]
pub fn svabs_s64_z(pg: svbool_t, op: svint64_t) -> svint64_t {
    svabs_s64_m(pg, svsel_s64(pg, op, svdup_n_s64(0)))
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_s8_m(inactive: svint8_t, pg: svbool_t, op: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cnot.nxv16i8")]
        fn _svcnot_s8_m(inactive: svint8_t, pg: svbool_t, op: svint8_t) -> svint8_t;
    }
    unsafe { _svcnot_s8_m(inactive, pg, op) }
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_s8_x(pg: svbool_t, op: svint8_t) -> svint8_t {
    svcnot_s8_m(op, pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_s8_z(pg: svbool_t, op: svint8_t) -> svint8_t {
    svcnot_s8_m(svdup_n_s8(0), pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_s16_m(inactive: svint16_t, pg: svbool_t, op: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cnot.nxv8i16")]
        fn _svcnot_s16_m(inactive: svint16_t, pg: svbool8_t, op: svint16_t) -> svint16_t;
    }
    unsafe { _svcnot_s16_m(inactive, simd_cast(pg), op) }
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_s16_x(pg: svbool_t, op: svint16_t) -> svint16_t {
    svcnot_s16_m(op, pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_s16_z(pg: svbool_t, op: svint16_t) -> svint16_t {
    svcnot_s16_m(svdup_n_s16(0), pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_s32_m(inactive: svint32_t, pg: svbool_t, op: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cnot.nxv4i32")]
        fn _svcnot_s32_m(inactive: svint32_t, pg: svbool4_t, op: svint32_t) -> svint32_t;
    }
    unsafe { _svcnot_s32_m(inactive, simd_cast(pg), op) }
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_s32_x(pg: svbool_t, op: svint32_t) -> svint32_t {
    svcnot_s32_m(op, pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_s32_z(pg: svbool_t, op: svint32_t) -> svint32_t {
    svcnot_s32_m(svdup_n_s32(0), pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_s64_m(inactive: svint64_t, pg: svbool_t, op: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cnot.nxv2i64")]
        fn _svcnot_s64_m(inactive: svint64_t, pg: svbool2_t, op: svint64_t) -> svint64_t;
    }
    unsafe { _svcnot_s64_m(inactive, simd_cast(pg), op) }
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_s64_x(pg: svbool_t, op: svint64_t) -> svint64_t {
    svcnot_s64_m(op, pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_s64_z(pg: svbool_t, op: svint64_t) -> svint64_t {
    svcnot_s64_m(svdup_n_s64(0), pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_u8_m(inactive: svuint8_t, pg: svbool_t, op: svuint8_t) -> svuint8_t {
    unsafe { svcnot_s8_m(inactive.as_signed(), pg, op.as_signed()).as_unsigned() }
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_u8_x(pg: svbool_t, op: svuint8_t) -> svuint8_t {
    svcnot_u8_m(op, pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_u8_z(pg: svbool_t, op: svuint8_t) -> svuint8_t {
    svcnot_u8_m(svdup_n_u8(0), pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_u16_m(inactive: svuint16_t, pg: svbool_t, op: svuint16_t) -> svuint16_t {
    unsafe { svcnot_s16_m(inactive.as_signed(), pg, op.as_signed()).as_unsigned() }
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_u16_x(pg: svbool_t, op: svuint16_t) -> svuint16_t {
    svcnot_u16_m(op, pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_u16_z(pg: svbool_t, op: svuint16_t) -> svuint16_t {
    svcnot_u16_m(svdup_n_u16(0), pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_u32_m(inactive: svuint32_t, pg: svbool_t, op: svuint32_t) -> svuint32_t {
    unsafe { svcnot_s32_m(inactive.as_signed(), pg, op.as_signed()).as_unsigned() }
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_u32_x(pg: svbool_t, op: svuint32_t) -> svuint32_t {
    svcnot_u32_m(op, pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_u32_z(pg: svbool_t, op: svuint32_t) -> svuint32_t {
    svcnot_u32_m(svdup_n_u32(0), pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_u64_m(inactive: svuint64_t, pg: svbool_t, op: svuint64_t) -> svuint64_t {
    unsafe { svcnot_s64_m(inactive.as_signed(), pg, op.as_signed()).as_unsigned() }
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_u64_x(pg: svbool_t, op: svuint64_t) -> svuint64_t {
    svcnot_u64_m(op, pg, op)
}
#[doc = "Conditional bitwise NOT"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnot[_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnot))]
pub fn svcnot_u64_z(pg: svbool_t, op: svuint64_t) -> svuint64_t {
    svcnot_u64_m(svdup_n_u64(0), pg, op)
}
// ============================================================================
// Batch 3: Reduction/Horizontal Operations
// ============================================================================
#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaddv[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(addv))]
pub fn svaddv_s8(pg: svbool_t, op: svint8_t) -> i64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.addv.nxv16i8")]
        fn _svaddv_s8(pg: svbool8_t, op: svint8_t) -> i64;
    }
    unsafe { _svaddv_s8(simd_cast(pg), op) }
}
#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaddv[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(addv))]
pub fn svaddv_s16(pg: svbool_t, op: svint16_t) -> i64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.addv.nxv8i16")]
        fn _svaddv_s16(pg: svbool4_t, op: svint16_t) -> i64;
    }
    unsafe { _svaddv_s16(simd_cast(pg), op) }
}
#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaddv[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(addv))]
pub fn svaddv_s32(pg: svbool_t, op: svint32_t) -> i64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.addv.nxv4i32")]
        fn _svaddv_s32(pg: svbool2_t, op: svint32_t) -> i64;
    }
    unsafe { _svaddv_s32(simd_cast(pg), op) }
}
#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaddv[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(addv))]
pub fn svaddv_s64(pg: svbool_t, op: svint64_t) -> i64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.addv.nxv2i64")]
        fn _svaddv_s64(pg: svbool_t, op: svint64_t) -> i64;
    }
    unsafe { _svaddv_s64(pg, op) }
}
#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaddv[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(addv))]
pub fn svaddv_u8(pg: svbool_t, op: svuint8_t) -> u64 {
    unsafe { svaddv_s8(pg, op.as_signed()) as u64 }
}
#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaddv[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(addv))]
pub fn svaddv_u16(pg: svbool_t, op: svuint16_t) -> u64 {
    unsafe { svaddv_s16(pg, op.as_signed()) as u64 }
}
#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaddv[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(addv))]
pub fn svaddv_u32(pg: svbool_t, op: svuint32_t) -> u64 {
    unsafe { svaddv_s32(pg, op.as_signed()) as u64 }
}
#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaddv[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(addv))]
pub fn svaddv_u64(pg: svbool_t, op: svuint64_t) -> u64 {
    unsafe { svaddv_s64(pg, op.as_signed()) as u64 }
}
#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaddv[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(faddv))]
pub fn svaddv_f32(pg: svbool_t, op: svfloat32_t) -> f32 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.faddv.nxv4f32")]
        fn _svaddv_f32(pg: svbool4_t, op: svfloat32_t) -> f32;
    }
    unsafe { _svaddv_f32(simd_cast(pg), op) }
}
#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svaddv[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(faddv))]
pub fn svaddv_f64(pg: svbool_t, op: svfloat64_t) -> f64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.faddv.nxv2f64")]
        fn _svaddv_f64(pg: svbool2_t, op: svfloat64_t) -> f64;
    }
    unsafe { _svaddv_f64(simd_cast(pg), op) }
}
#[doc = "Count active predicate elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcntb)]"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cntb))]
pub fn svcntb() -> i32 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cntb")]
        fn _svcntb() -> i32;
    }
    unsafe { _svcntb() }
}
#[doc = "Count active predicate elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcnth)]"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cnth))]
pub fn svcnth() -> i32 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cnth")]
        fn _svcnth() -> i32;
    }
    unsafe { _svcnth() }
}
#[doc = "Count active predicate elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcntd)]"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cntd))]
pub fn svcntd() -> i32 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cntd")]
        fn _svcntd() -> i32;
    }
    unsafe { _svcntd() }
}
#[doc = "Count active predicate elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcntp[_b8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cntp))]
pub fn svcntp_b8(pg: svbool_t, op: svbool_t) -> u64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cntp.nxv16i1")]
        fn _svcntp_b8(pg: svbool8_t, op: svbool8_t) -> u64;
    }
    unsafe { _svcntp_b8(simd_cast(pg), simd_cast(op)) }
}
#[doc = "Count active predicate elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcntp[_b16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cntp))]
pub fn svcntp_b16(pg: svbool_t, op: svbool_t) -> u64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cntp.nxv8i1")]
        fn _svcntp_b16(pg: svbool4_t, op: svbool4_t) -> u64;
    }
    unsafe { _svcntp_b16(simd_cast(pg), simd_cast(op)) }
}
#[doc = "Count active predicate elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcntp[_b32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cntp))]
pub fn svcntp_b32(pg: svbool_t, op: svbool_t) -> u64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cntp.nxv4i1")]
        fn _svcntp_b32(pg: svbool2_t, op: svbool2_t) -> u64;
    }
    unsafe { _svcntp_b32(simd_cast(pg), simd_cast(op)) }
}
#[doc = "Count active predicate elements"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcntp[_b64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cntp))]
pub fn svcntp_b64(pg: svbool_t, op: svbool_t) -> u64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cntp.nxv2i1")]
        fn _svcntp_b64(pg: svbool_t, op: svbool_t) -> u64;
    }
    unsafe { _svcntp_b64(pg, op) }
}
#[doc = "Count leading zeros"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svclz[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(clz))]
pub fn svclz_s8(pg: svbool_t, op: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.clz.nxv16i8")]
        fn _svclz_s8(pg: svbool8_t, op: svint8_t) -> svint8_t;
    }
    unsafe { _svclz_s8(simd_cast(pg), op) }
}
#[doc = "Count leading zeros"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svclz[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(clz))]
pub fn svclz_s16(pg: svbool_t, op: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.clz.nxv8i16")]
        fn _svclz_s16(pg: svbool4_t, op: svint16_t) -> svint16_t;
    }
    unsafe { _svclz_s16(simd_cast(pg), op) }
}
#[doc = "Count leading zeros"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svclz[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(clz))]
pub fn svclz_s32(pg: svbool_t, op: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.clz.nxv4i32")]
        fn _svclz_s32(pg: svbool2_t, op: svint32_t) -> svint32_t;
    }
    unsafe { _svclz_s32(simd_cast(pg), op) }
}
#[doc = "Count leading zeros"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svclz[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(clz))]
pub fn svclz_s64(pg: svbool_t, op: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.clz.nxv2i64")]
        fn _svclz_s64(pg: svbool_t, op: svint64_t) -> svint64_t;
    }
    unsafe { _svclz_s64(pg, op) }
}
#[doc = "Count leading zeros"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svclz[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(clz))]
pub fn svclz_u8(pg: svbool_t, op: svuint8_t) -> svuint8_t {
    unsafe { svclz_s8(pg, op.as_signed()).as_unsigned() }
}
#[doc = "Count leading zeros"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svclz[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(clz))]
pub fn svclz_u16(pg: svbool_t, op: svuint16_t) -> svuint16_t {
    unsafe { svclz_s16(pg, op.as_signed()).as_unsigned() }
}
#[doc = "Count leading zeros"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svclz[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(clz))]
pub fn svclz_u32(pg: svbool_t, op: svuint32_t) -> svuint32_t {
    unsafe { svclz_s32(pg, op.as_signed()).as_unsigned() }
}
#[doc = "Count leading zeros"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svclz[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(clz))]
pub fn svclz_u64(pg: svbool_t, op: svuint64_t) -> svuint64_t {
    unsafe { svclz_s64(pg, op.as_signed()).as_unsigned() }
}
#[doc = "Count leading sign bits"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcls[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cls))]
pub fn svcls_s8(pg: svbool_t, op: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cls.nxv16i8")]
        fn _svcls_s8(pg: svbool8_t, op: svint8_t) -> svint8_t;
    }
    unsafe { _svcls_s8(simd_cast(pg), op) }
}
#[doc = "Count leading sign bits"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcls[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cls))]
pub fn svcls_s16(pg: svbool_t, op: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cls.nxv8i16")]
        fn _svcls_s16(pg: svbool4_t, op: svint16_t) -> svint16_t;
    }
    unsafe { _svcls_s16(simd_cast(pg), op) }
}
#[doc = "Count leading sign bits"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcls[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cls))]
pub fn svcls_s32(pg: svbool_t, op: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cls.nxv4i32")]
        fn _svcls_s32(pg: svbool2_t, op: svint32_t) -> svint32_t;
    }
    unsafe { _svcls_s32(simd_cast(pg), op) }
}
#[doc = "Count leading sign bits"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcls[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cls))]
pub fn svcls_s64(pg: svbool_t, op: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.cls.nxv2i64")]
        fn _svcls_s64(pg: svbool_t, op: svint64_t) -> svint64_t;
    }
    unsafe { _svcls_s64(pg, op) }
}
#[doc = "Count leading sign bits"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcls[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cls))]
pub fn svcls_u8(pg: svbool_t, op: svuint8_t) -> svuint8_t {
    unsafe { svcls_s8(pg, op.as_signed()).as_unsigned() }
}
#[doc = "Count leading sign bits"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcls[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cls))]
pub fn svcls_u16(pg: svbool_t, op: svuint16_t) -> svuint16_t {
    unsafe { svcls_s16(pg, op.as_signed()).as_unsigned() }
}
#[doc = "Count leading sign bits"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcls[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cls))]
pub fn svcls_u32(pg: svbool_t, op: svuint32_t) -> svuint32_t {
    unsafe { svcls_s32(pg, op.as_signed()).as_unsigned() }
}
#[doc = "Count leading sign bits"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcls[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(cls))]
pub fn svcls_u64(pg: svbool_t, op: svuint64_t) -> svuint64_t {
    unsafe { svcls_s64(pg, op.as_signed()).as_unsigned() }
}

// ============================================================================
// 第4批：地址与加载/存储族 Intrinsics
// ============================================================================

// ----------------------------------------------------------------------------
// svadr - 地址生成函数
// ----------------------------------------------------------------------------

#[doc = "Address generation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadr[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(adr))]
pub unsafe fn svadr_s32(pg: svbool_t, base: *const i8, offset: svint32_t) -> svuint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.adr.nxv4i32")]
        fn _svadr_s32(pg: svbool4_t, base: *const i8, offset: svint32_t) -> svuint64_t;
    }
    _svadr_s32(simd_cast(pg), base, offset)
}

#[doc = "Address generation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadr[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(adr))]
pub unsafe fn svadr_s64(pg: svbool_t, base: *const i8, offset: svint64_t) -> svuint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.adr.nxv2i64")]
        fn _svadr_s64(pg: svbool2_t, base: *const i8, offset: svint64_t) -> svuint64_t;
    }
    _svadr_s64(simd_cast(pg), base, offset)
}

#[doc = "Address generation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadr[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(adr))]
pub unsafe fn svadr_u32(pg: svbool_t, base: *const i8, offset: svuint32_t) -> svuint64_t {
    unsafe { svadr_s32(pg, base, offset.as_signed()).as_unsigned() }
}

#[doc = "Address generation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadr[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(adr))]
pub unsafe fn svadr_u64(pg: svbool_t, base: *const i8, offset: svuint64_t) -> svuint64_t {
    unsafe { svadr_s64(pg, base, offset.as_signed()).as_unsigned() }
}

// ----------------------------------------------------------------------------
// svld1_vnum - 带向量索引的加载
// ----------------------------------------------------------------------------

#[doc = "Unextended load (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_vnum[_f32])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1w))]
pub unsafe fn svld1_vnum_f32(pg: svbool_t, base: *const f32, vnum: i64) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.nxv4f32")]
        fn _svld1_vnum_f32(pg: svbool4_t, base: *const f32) -> svfloat32_t;
    }
    let offset_base = base.add(vnum as usize * 4);
    _svld1_vnum_f32(simd_cast(pg), offset_base)
}

#[doc = "Unextended load (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_vnum[_f64])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1d))]
pub unsafe fn svld1_vnum_f64(pg: svbool_t, base: *const f64, vnum: i64) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.nxv2f64")]
        fn _svld1_vnum_f64(pg: svbool2_t, base: *const f64) -> svfloat64_t;
    }
    let offset_base = base.add(vnum as usize * 2);
    _svld1_vnum_f64(simd_cast(pg), offset_base)
}

#[doc = "Unextended load (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_vnum[_s8])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1b))]
pub unsafe fn svld1_vnum_s8(pg: svbool_t, base: *const i8, vnum: i64) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.nxv16i8")]
        fn _svld1_vnum_s8(pg: svbool_t, base: *const i8) -> svint8_t;
    }
    let offset_base = base.add(vnum as usize * 16);
    _svld1_vnum_s8(pg, offset_base)
}

#[doc = "Unextended load (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_vnum[_s16])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1h))]
pub unsafe fn svld1_vnum_s16(pg: svbool_t, base: *const i16, vnum: i64) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.nxv8i16")]
        fn _svld1_vnum_s16(pg: svbool8_t, base: *const i16) -> svint16_t;
    }
    let offset_base = base.add(vnum as usize * 8);
    _svld1_vnum_s16(simd_cast(pg), offset_base)
}

#[doc = "Unextended load (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_vnum[_s32])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1w))]
pub unsafe fn svld1_vnum_s32(pg: svbool_t, base: *const i32, vnum: i64) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.nxv4i32")]
        fn _svld1_vnum_s32(pg: svbool4_t, base: *const i32) -> svint32_t;
    }
    let offset_base = base.add(vnum as usize * 4);
    _svld1_vnum_s32(simd_cast(pg), offset_base)
}

#[doc = "Unextended load (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_vnum[_s64])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1d))]
pub unsafe fn svld1_vnum_s64(pg: svbool_t, base: *const i64, vnum: i64) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.nxv2i64")]
        fn _svld1_vnum_s64(pg: svbool2_t, base: *const i64) -> svint64_t;
    }
    let offset_base = base.add(vnum as usize * 2);
    _svld1_vnum_s64(simd_cast(pg), offset_base)
}

#[doc = "Unextended load (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_vnum[_u8])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1b))]
pub unsafe fn svld1_vnum_u8(pg: svbool_t, base: *const u8, vnum: i64) -> svuint8_t {
    svld1_vnum_s8(pg, base.as_signed(), vnum).as_unsigned()
}

#[doc = "Unextended load (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_vnum[_u16])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1h))]
pub unsafe fn svld1_vnum_u16(pg: svbool_t, base: *const u16, vnum: i64) -> svuint16_t {
    svld1_vnum_s16(pg, base.as_signed(), vnum).as_unsigned()
}

#[doc = "Unextended load (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_vnum[_u32])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1w))]
pub unsafe fn svld1_vnum_u32(pg: svbool_t, base: *const u32, vnum: i64) -> svuint32_t {
    svld1_vnum_s32(pg, base.as_signed(), vnum).as_unsigned()
}

#[doc = "Unextended load (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_vnum[_u64])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1d))]
pub unsafe fn svld1_vnum_u64(pg: svbool_t, base: *const u64, vnum: i64) -> svuint64_t {
    svld1_vnum_s64(pg, base.as_signed(), vnum).as_unsigned()
}

// ----------------------------------------------------------------------------
// svld1_gather - 聚集加载
// ----------------------------------------------------------------------------

#[doc = "Gather load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_gather[_s32index]_f32)"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1w))]
pub unsafe fn svld1_gather_s32index_f32(
    pg: svbool_t,
    base: *const f32,
    indices: svint32_t,
) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.gather.index.nxv4f32")]
        fn _svld1_gather_s32index_f32(pg: svbool4_t, base: *const f32, indices: svint32_t) -> svfloat32_t;
    }
    _svld1_gather_s32index_f32(simd_cast(pg), base, indices)
}

#[doc = "Gather load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_gather[_s64index]_f64)"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1d))]
pub unsafe fn svld1_gather_s64index_f64(
    pg: svbool_t,
    base: *const f64,
    indices: svint64_t,
) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.gather.index.nxv2f64")]
        fn _svld1_gather_s64index_f64(pg: svbool2_t, base: *const f64, indices: svint64_t) -> svfloat64_t;
    }
    _svld1_gather_s64index_f64(simd_cast(pg), base, indices)
}

#[doc = "Gather load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_gather[_s32index]_s32)"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1w))]
pub unsafe fn svld1_gather_s32index_s32(
    pg: svbool_t,
    base: *const i32,
    indices: svint32_t,
) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.gather.index.nxv4i32")]
        fn _svld1_gather_s32index_s32(pg: svbool4_t, base: *const i32, indices: svint32_t) -> svint32_t;
    }
    _svld1_gather_s32index_s32(simd_cast(pg), base, indices)
}

#[doc = "Gather load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_gather[_s64index]_s64)"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1d))]
pub unsafe fn svld1_gather_s64index_s64(
    pg: svbool_t,
    base: *const i64,
    indices: svint64_t,
) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.ld1.gather.index.nxv2i64")]
        fn _svld1_gather_s64index_s64(pg: svbool2_t, base: *const i64, indices: svint64_t) -> svint64_t;
    }
    _svld1_gather_s64index_s64(simd_cast(pg), base, indices)
}

#[doc = "Gather load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_gather[_u32index]_u32)"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1w))]
pub unsafe fn svld1_gather_u32index_u32(
    pg: svbool_t,
    base: *const u32,
    indices: svuint32_t,
) -> svuint32_t {
    unsafe {
        svld1_gather_s32index_s32(pg, base.as_signed(), indices.as_signed()).as_unsigned()
    }
}

#[doc = "Gather load"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svld1_gather[_u64index]_u64)"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(ld1d))]
pub unsafe fn svld1_gather_u64index_u64(
    pg: svbool_t,
    base: *const u64,
    indices: svuint64_t,
) -> svuint64_t {
    unsafe {
        svld1_gather_s64index_s64(pg, base.as_signed(), indices.as_signed()).as_unsigned()
    }
}

// ----------------------------------------------------------------------------
// svst1_vnum - 带向量索引的存储
// ----------------------------------------------------------------------------

#[doc = "Unextended store (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_vnum[_f32])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1w))]
pub unsafe fn svst1_vnum_f32(pg: svbool_t, base: *mut f32, vnum: i64, data: svfloat32_t) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.nxv4f32")]
        fn _svst1_vnum_f32(data: svfloat32_t, pg: svbool4_t, ptr: *mut f32);
    }
    let offset_base = base.add(vnum as usize * 4);
    _svst1_vnum_f32(data, simd_cast(pg), offset_base)
}

#[doc = "Unextended store (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_vnum[_f64])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1d))]
pub unsafe fn svst1_vnum_f64(pg: svbool_t, base: *mut f64, vnum: i64, data: svfloat64_t) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.nxv2f64")]
        fn _svst1_vnum_f64(data: svfloat64_t, pg: svbool2_t, ptr: *mut f64);
    }
    let offset_base = base.add(vnum as usize * 2);
    _svst1_vnum_f64(data, simd_cast(pg), offset_base)
}

#[doc = "Unextended store (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_vnum[_s8])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1b))]
pub unsafe fn svst1_vnum_s8(pg: svbool_t, base: *mut i8, vnum: i64, data: svint8_t) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.nxv16i8")]
        fn _svst1_vnum_s8(data: svint8_t, pg: svbool_t, ptr: *mut i8);
    }
    let offset_base = base.add(vnum as usize * 16);
    _svst1_vnum_s8(data, pg, offset_base)
}

#[doc = "Unextended store (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_vnum[_s16])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1h))]
pub unsafe fn svst1_vnum_s16(pg: svbool_t, base: *mut i16, vnum: i64, data: svint16_t) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.nxv8i16")]
        fn _svst1_vnum_s16(data: svint16_t, pg: svbool8_t, ptr: *mut i16);
    }
    let offset_base = base.add(vnum as usize * 8);
    _svst1_vnum_s16(data, simd_cast(pg), offset_base)
}

#[doc = "Unextended store (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_vnum[_s32])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1w))]
pub unsafe fn svst1_vnum_s32(pg: svbool_t, base: *mut i32, vnum: i64, data: svint32_t) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.nxv4i32")]
        fn _svst1_vnum_s32(data: svint32_t, pg: svbool4_t, ptr: *mut i32);
    }
    let offset_base = base.add(vnum as usize * 4);
    _svst1_vnum_s32(data, simd_cast(pg), offset_base)
}

#[doc = "Unextended store (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_vnum[_s64])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1d))]
pub unsafe fn svst1_vnum_s64(pg: svbool_t, base: *mut i64, vnum: i64, data: svint64_t) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.nxv2i64")]
        fn _svst1_vnum_s64(data: svint64_t, pg: svbool2_t, ptr: *mut i64);
    }
    let offset_base = base.add(vnum as usize * 2);
    _svst1_vnum_s64(data, simd_cast(pg), offset_base)
}

#[doc = "Unextended store (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_vnum[_u8])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1b))]
pub unsafe fn svst1_vnum_u8(pg: svbool_t, base: *mut u8, vnum: i64, data: svuint8_t) {
    svst1_vnum_s8(pg, base.as_signed(), vnum, data.as_signed())
}

#[doc = "Unextended store (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_vnum[_u16])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1h))]
pub unsafe fn svst1_vnum_u16(pg: svbool_t, base: *mut u16, vnum: i64, data: svuint16_t) {
    svst1_vnum_s16(pg, base.as_signed(), vnum, data.as_signed())
}

#[doc = "Unextended store (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_vnum[_u32])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1w))]
pub unsafe fn svst1_vnum_u32(pg: svbool_t, base: *mut u32, vnum: i64, data: svuint32_t) {
    svst1_vnum_s32(pg, base.as_signed(), vnum, data.as_signed())
}

#[doc = "Unextended store (vector base + scalar offset)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_vnum[_u64])"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1d))]
pub unsafe fn svst1_vnum_u64(pg: svbool_t, base: *mut u64, vnum: i64, data: svuint64_t) {
    svst1_vnum_s64(pg, base.as_signed(), vnum, data.as_signed())
}

// ----------------------------------------------------------------------------
// svst1_scatter - 分散存储
// ----------------------------------------------------------------------------

#[doc = "Scatter store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_scatter[_s32index]_f32)"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1w))]
pub unsafe fn svst1_scatter_s32index_f32(
    pg: svbool_t,
    base: *mut f32,
    indices: svint32_t,
    data: svfloat32_t,
) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.scatter.index.nxv4f32")]
        fn _svst1_scatter_s32index_f32(data: svfloat32_t, pg: svbool4_t, base: *mut f32, indices: svint32_t);
    }
    _svst1_scatter_s32index_f32(data, simd_cast(pg), base, indices)
}

#[doc = "Scatter store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_scatter[_s64index]_f64)"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1d))]
pub unsafe fn svst1_scatter_s64index_f64(
    pg: svbool_t,
    base: *mut f64,
    indices: svint64_t,
    data: svfloat64_t,
) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.scatter.index.nxv2f64")]
        fn _svst1_scatter_s64index_f64(data: svfloat64_t, pg: svbool2_t, base: *mut f64, indices: svint64_t);
    }
    _svst1_scatter_s64index_f64(data, simd_cast(pg), base, indices)
}

#[doc = "Scatter store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_scatter[_s32index]_s32)"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1w))]
pub unsafe fn svst1_scatter_s32index_s32(
    pg: svbool_t,
    base: *mut i32,
    indices: svint32_t,
    data: svint32_t,
) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.scatter.index.nxv4i32")]
        fn _svst1_scatter_s32index_s32(data: svint32_t, pg: svbool4_t, base: *mut i32, indices: svint32_t);
    }
    _svst1_scatter_s32index_s32(data, simd_cast(pg), base, indices)
}

#[doc = "Scatter store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_scatter[_s64index]_s64)"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1d))]
pub unsafe fn svst1_scatter_s64index_s64(
    pg: svbool_t,
    base: *mut i64,
    indices: svint64_t,
    data: svint64_t,
) {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.st1.scatter.index.nxv2i64")]
        fn _svst1_scatter_s64index_s64(data: svint64_t, pg: svbool2_t, base: *mut i64, indices: svint64_t);
    }
    _svst1_scatter_s64index_s64(data, simd_cast(pg), base, indices)
}

#[doc = "Scatter store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_scatter[_u32index]_u32)"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1w))]
pub unsafe fn svst1_scatter_u32index_u32(
    pg: svbool_t,
    base: *mut u32,
    indices: svuint32_t,
    data: svuint32_t,
) {
    unsafe {
        svst1_scatter_s32index_s32(pg, base.as_signed(), indices.as_signed(), data.as_signed())
    }
}

#[doc = "Scatter store"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svst1_scatter[_u64index]_u64)"]
#[doc = ""]
#[doc = "## Safety"]
#[doc = "  * [`pointer::offset`](pointer#method.offset) safety constraints must be met for the address calculation for each active element (governed by `pg`)."]
#[doc = "  * This dereferences and accesses the calculated address for each active element (governed by `pg`)."]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(st1d))]
pub unsafe fn svst1_scatter_u64index_u64(
    pg: svbool_t,
    base: *mut u64,
    indices: svuint64_t,
    data: svuint64_t,
) {
    unsafe {
        svst1_scatter_s64index_s64(pg, base.as_signed(), indices.as_signed(), data.as_signed())
    }
}

// ============================================================================
// Additional SVE intrinsics generated based on ARM documentation and test files
// ============================================================================

#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadda_f16)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadda_f16(pg: svbool_t, initial: f16, op: svfloat16_t) -> f16 {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.fadda.nxv8f16"
        )]
        fn _svadda_f16(pg: svbool8_t, initial: f16, op: svfloat16_t) -> f16;
    }
    unsafe { _svadda_f16(simd_cast(pg), initial, op) }
}
#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadda_f32)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadda_f32(pg: svbool_t, initial: f32, op: svfloat32_t) -> f32 {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.fadda.nxv4f32"
        )]
        fn _svadda_f32(pg: svbool4_t, initial: f32, op: svfloat32_t) -> f32;
    }
    unsafe { _svadda_f32(simd_cast(pg), initial, op) }
}
#[doc = "Add across vector"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadda_f64)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadda_f64(pg: svbool_t, initial: f64, op: svfloat64_t) -> f64 {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.fadda.nxv2f64"
        )]
        fn _svadda_f64(pg: svbool2_t, initial: f64, op: svfloat64_t) -> f64;
    }
    unsafe { _svadda_f64(simd_cast(pg), initial, op) }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrb_u32base_s32offset)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrb_u32base_s32offset(bases: svuint32_t, offsets: svint32_t) -> svuint32_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.adrb.nxv4i32"
        )]
        fn _svadrb_u32base_s32offset(bases: svint32_t, offsets: svint32_t) -> svint32_t;
    }
    unsafe { _svadrb_u32base_s32offset(bases.as_signed(), offsets).as_unsigned() }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrb_u32base_u32offset)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrb_u32base_u32offset(bases: svuint32_t, offsets: svuint32_t) -> svuint32_t {
    unsafe { svadrb_u32base_s32offset(bases, offsets.as_signed()) }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrb_u64base_s64offset)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrb_u64base_s64offset(bases: svuint64_t, offsets: svint64_t) -> svuint64_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.adrb.nxv2i64"
        )]
        fn _svadrb_u64base_s64offset(bases: svint64_t, offsets: svint64_t) -> svint64_t;
    }
    unsafe { _svadrb_u64base_s64offset(bases.as_signed(), offsets).as_unsigned() }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrb_u64base_u64offset)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrb_u64base_u64offset(bases: svuint64_t, offsets: svuint64_t) -> svuint64_t {
    unsafe { svadrb_u64base_s64offset(bases, offsets.as_signed()) }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrd_u32base_s32index)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrd_u32base_s32index(bases: svuint32_t, indices: svint32_t) -> svuint32_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.adrd.nxv4i32"
        )]
        fn _svadrd_u32base_s32index(bases: svint32_t, indices: svint32_t) -> svint32_t;
    }
    unsafe { _svadrd_u32base_s32index(bases.as_signed(), indices).as_unsigned() }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrd_u32base_u32index)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrd_u32base_u32index(bases: svuint32_t, indices: svuint32_t) -> svuint32_t {
    unsafe { svadrd_u32base_s32index(bases, indices.as_signed()) }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrd_u64base_s64index)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrd_u64base_s64index(bases: svuint64_t, indices: svint64_t) -> svuint64_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.adrd.nxv2i64"
        )]
        fn _svadrd_u64base_s64index(bases: svint64_t, indices: svint64_t) -> svint64_t;
    }
    unsafe { _svadrd_u64base_s64index(bases.as_signed(), indices).as_unsigned() }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrd_u64base_u64index)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrd_u64base_u64index(bases: svuint64_t, indices: svuint64_t) -> svuint64_t {
    unsafe { svadrd_u64base_s64index(bases, indices.as_signed()) }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrh_u32base_s32index)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrh_u32base_s32index(bases: svuint32_t, indices: svint32_t) -> svuint32_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.adrh.nxv4i32"
        )]
        fn _svadrh_u32base_s32index(bases: svint32_t, indices: svint32_t) -> svint32_t;
    }
    unsafe { _svadrh_u32base_s32index(bases.as_signed(), indices).as_unsigned() }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrh_u32base_u32index)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrh_u32base_u32index(bases: svuint32_t, indices: svuint32_t) -> svuint32_t {
    unsafe { svadrh_u32base_s32index(bases, indices.as_signed()) }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrh_u64base_s64index)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrh_u64base_s64index(bases: svuint64_t, indices: svint64_t) -> svuint64_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.adrh.nxv2i64"
        )]
        fn _svadrh_u64base_s64index(bases: svint64_t, indices: svint64_t) -> svint64_t;
    }
    unsafe { _svadrh_u64base_s64index(bases.as_signed(), indices).as_unsigned() }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrh_u64base_u64index)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrh_u64base_u64index(bases: svuint64_t, indices: svuint64_t) -> svuint64_t {
    unsafe { svadrh_u64base_s64index(bases, indices.as_signed()) }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrw_u32base_s32index)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrw_u32base_s32index(bases: svuint32_t, indices: svint32_t) -> svuint32_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.adrw.nxv4i32"
        )]
        fn _svadrw_u32base_s32index(bases: svint32_t, indices: svint32_t) -> svint32_t;
    }
    unsafe { _svadrw_u32base_s32index(bases.as_signed(), indices).as_unsigned() }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrw_u32base_u32index)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrw_u32base_u32index(bases: svuint32_t, indices: svuint32_t) -> svuint32_t {
    unsafe { svadrw_u32base_s32index(bases, indices.as_signed()) }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrw_u64base_s64index)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrw_u64base_s64index(bases: svuint64_t, indices: svint64_t) -> svuint64_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.adrw.nxv2i64"
        )]
        fn _svadrw_u64base_s64index(bases: svint64_t, indices: svint64_t) -> svint64_t;
    }
    unsafe { _svadrw_u64base_s64index(bases.as_signed(), indices).as_unsigned() }
}
#[doc = "Address calculation"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svadrw_u64base_u64index)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svadrw_u64base_u64index(bases: svuint64_t, indices: svuint64_t) -> svuint64_t {
    unsafe { svadrw_u64base_s64index(bases, indices.as_signed()) }
}
#[doc = "Compare equal (wide)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq_wide_s8)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svcmpeq_wide_s8(pg: svbool_t, op1: svint8_t, op2: svint64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.cmpeq.wide.nxv16i8"
        )]
        fn _svcmpeq_wide_s8(pg: svbool_t, op1: svint8_t, op2: svint64_t) -> svbool_t;
    }
    unsafe { _svcmpeq_wide_s8(pg, op1, op2) }
}
#[doc = "Compare equal (wide)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq_wide_s16)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svcmpeq_wide_s16(pg: svbool_t, op1: svint16_t, op2: svint64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.cmpeq.wide.nxv8i16"
        )]
        fn _svcmpeq_wide_s16(pg: svbool8_t, op1: svint16_t, op2: svint64_t) -> svbool8_t;
    }
    unsafe { simd_cast(_svcmpeq_wide_s16(simd_cast(pg), op1, op2)) }
}
#[doc = "Compare equal (wide)"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcmpeq_wide_s32)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub unsafe fn svcmpeq_wide_s32(pg: svbool_t, op1: svint32_t, op2: svint64_t) -> svbool_t {
    unsafe extern "C" {
        #[cfg_attr(
            target_arch = "aarch64",
            link_name = "llvm.aarch64.sve.cmpeq.wide.nxv4i32"
        )]
        fn _svcmpeq_wide_s32(pg: svbool4_t, op1: svint32_t, op2: svint64_t) -> svbool4_t;
    }
    unsafe { simd_cast(_svcmpeq_wide_s32(simd_cast(pg), op1, op2)) }
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqsub))]
pub fn svqsub_s8(op1: svint8_t, op2: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sqsub.x.nxv16i8")]
        fn _svqsub_s8(op1: svint8_t, op2: svint8_t) -> svint8_t;
    }
    unsafe { _svqsub_s8(op1, op2) }
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_n_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqsub))]
pub fn svqsub_n_s8(op1: svint8_t, op2: i8) -> svint8_t {
    svqsub_s8(op1, svdup_n_s8(op2))
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqsub))]
pub fn svqsub_s16(op1: svint16_t, op2: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sqsub.x.nxv8i16")]
        fn _svqsub_s16(op1: svint16_t, op2: svint16_t) -> svint16_t;
    }
    unsafe { _svqsub_s16(op1, op2) }
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_n_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqsub))]
pub fn svqsub_n_s16(op1: svint16_t, op2: i16) -> svint16_t {
    svqsub_s16(op1, svdup_n_s16(op2))
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqsub))]
pub fn svqsub_s32(op1: svint32_t, op2: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sqsub.x.nxv4i32")]
        fn _svqsub_s32(op1: svint32_t, op2: svint32_t) -> svint32_t;
    }
    unsafe { _svqsub_s32(op1, op2) }
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_n_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqsub))]
pub fn svqsub_n_s32(op1: svint32_t, op2: i32) -> svint32_t {
    svqsub_s32(op1, svdup_n_s32(op2))
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqsub))]
pub fn svqsub_s64(op1: svint64_t, op2: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sqsub.x.nxv2i64")]
        fn _svqsub_s64(op1: svint64_t, op2: svint64_t) -> svint64_t;
    }
    unsafe { _svqsub_s64(op1, op2) }
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_n_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqsub))]
pub fn svqsub_n_s64(op1: svint64_t, op2: i64) -> svint64_t {
    svqsub_s64(op1, svdup_n_s64(op2))
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqsub))]
pub fn svqsub_u8(op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.uqsub.x.nxv16i8")]
        fn _svqsub_u8(op1: svuint8_t, op2: svuint8_t) -> svuint8_t;
    }
    unsafe { _svqsub_u8(op1, op2) }
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_n_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqsub))]
pub fn svqsub_n_u8(op1: svuint8_t, op2: u8) -> svuint8_t {
    svqsub_u8(op1, svdup_n_u8(op2))
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqsub))]
pub fn svqsub_u16(op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.uqsub.x.nxv8i16")]
        fn _svqsub_u16(op1: svuint16_t, op2: svuint16_t) -> svuint16_t;
    }
    unsafe { _svqsub_u16(op1, op2) }
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_n_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqsub))]
pub fn svqsub_n_u16(op1: svuint16_t, op2: u16) -> svuint16_t {
    svqsub_u16(op1, svdup_n_u16(op2))
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqsub))]
pub fn svqsub_u32(op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.uqsub.x.nxv4i32")]
        fn _svqsub_u32(op1: svuint32_t, op2: svuint32_t) -> svuint32_t;
    }
    unsafe { _svqsub_u32(op1, op2) }
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_n_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqsub))]
pub fn svqsub_n_u32(op1: svuint32_t, op2: u32) -> svuint32_t {
    svqsub_u32(op1, svdup_n_u32(op2))
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqsub))]
pub fn svqsub_u64(op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.uqsub.x.nxv2i64")]
        fn _svqsub_u64(op1: svuint64_t, op2: svuint64_t) -> svuint64_t;
    }
    unsafe { _svqsub_u64(op1, op2) }
}
#[doc = "Saturating subtract"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqsub[_n_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqsub))]
pub fn svqsub_n_u64(op1: svuint64_t, op2: u64) -> svuint64_t {
    svqsub_u64(op1, svdup_n_u64(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsubr))]
pub fn svsubr_f32_m(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fsubr.nxv4f32")]
        fn _svsubr_f32_m(pg: svbool4_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t;
    }
    unsafe { _svsubr_f32_m(simd_cast(pg), op1, op2) }
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_f32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsubr))]
pub fn svsubr_n_f32_m(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svsubr_f32_m(pg, op1, svdup_n_f32(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsubr))]
pub fn svsubr_f32_x(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    svsubr_f32_m(pg, op1, op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_f32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsubr))]
pub fn svsubr_n_f32_x(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svsubr_f32_x(pg, op1, svdup_n_f32(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsubr))]
pub fn svsubr_f32_z(pg: svbool_t, op1: svfloat32_t, op2: svfloat32_t) -> svfloat32_t {
    svsubr_f32_m(pg, svsel_f32(pg, op1, svdup_n_f32(0.0)), op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_f32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsubr))]
pub fn svsubr_n_f32_z(pg: svbool_t, op1: svfloat32_t, op2: f32) -> svfloat32_t {
    svsubr_f32_z(pg, op1, svdup_n_f32(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsubr))]
pub fn svsubr_f64_m(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.fsubr.nxv2f64")]
        fn _svsubr_f64_m(pg: svbool2_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t;
    }
    unsafe { _svsubr_f64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_f64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsubr))]
pub fn svsubr_n_f64_m(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svsubr_f64_m(pg, op1, svdup_n_f64(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsubr))]
pub fn svsubr_f64_x(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    svsubr_f64_m(pg, op1, op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_f64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsubr))]
pub fn svsubr_n_f64_x(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svsubr_f64_x(pg, op1, svdup_n_f64(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsubr))]
pub fn svsubr_f64_z(pg: svbool_t, op1: svfloat64_t, op2: svfloat64_t) -> svfloat64_t {
    svsubr_f64_m(pg, svsel_f64(pg, op1, svdup_n_f64(0.0)), op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_f64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(fsubr))]
pub fn svsubr_n_f64_z(pg: svbool_t, op1: svfloat64_t, op2: f64) -> svfloat64_t {
    svsubr_f64_z(pg, op1, svdup_n_f64(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.subr.nxv16i8")]
        fn _svsubr_s8_m(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t;
    }
    unsafe { _svsubr_s8_m(pg, op1, op2) }
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_s8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_s8_m(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svsubr_s8_m(pg, op1, svdup_n_s8(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_s8_x(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svsubr_s8_m(pg, op1, op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_s8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_s8_x(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svsubr_s8_x(pg, op1, svdup_n_s8(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_s8_z(pg: svbool_t, op1: svint8_t, op2: svint8_t) -> svint8_t {
    svsubr_s8_m(pg, svsel_s8(pg, op1, svdup_n_s8(0)), op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_s8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_s8_z(pg: svbool_t, op1: svint8_t, op2: i8) -> svint8_t {
    svsubr_s8_z(pg, op1, svdup_n_s8(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_s16_m(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.subr.nxv8i16")]
        fn _svsubr_s16_m(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t;
    }
    unsafe { _svsubr_s16_m(pg, op1, op2) }
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_s16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_s16_m(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svsubr_s16_m(pg, op1, svdup_n_s16(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_s16_x(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svsubr_s16_m(pg, op1, op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_s16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_s16_x(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svsubr_s16_x(pg, op1, svdup_n_s16(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_s16_z(pg: svbool_t, op1: svint16_t, op2: svint16_t) -> svint16_t {
    svsubr_s16_m(pg, svsel_s16(pg, op1, svdup_n_s16(0)), op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_s16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_s16_z(pg: svbool_t, op1: svint16_t, op2: i16) -> svint16_t {
    svsubr_s16_z(pg, op1, svdup_n_s16(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_s32_m(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.subr.nxv4i32")]
        fn _svsubr_s32_m(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t;
    }
    unsafe { _svsubr_s32_m(pg, op1, op2) }
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_s32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_s32_m(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svsubr_s32_m(pg, op1, svdup_n_s32(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_s32_x(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svsubr_s32_m(pg, op1, op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_s32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_s32_x(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svsubr_s32_x(pg, op1, svdup_n_s32(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_s32_z(pg: svbool_t, op1: svint32_t, op2: svint32_t) -> svint32_t {
    svsubr_s32_m(pg, svsel_s32(pg, op1, svdup_n_s32(0)), op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_s32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_s32_z(pg: svbool_t, op1: svint32_t, op2: i32) -> svint32_t {
    svsubr_s32_z(pg, op1, svdup_n_s32(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_s64_m(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.subr.nxv2i64")]
        fn _svsubr_s64_m(pg: svbool2_t, op1: svint64_t, op2: svint64_t) -> svint64_t;
    }
    unsafe { _svsubr_s64_m(simd_cast(pg), op1, op2) }
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_s64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_s64_m(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svsubr_s64_m(pg, op1, svdup_n_s64(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_s64_x(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svsubr_s64_m(pg, op1, op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_s64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_s64_x(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svsubr_s64_x(pg, op1, svdup_n_s64(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_s64_z(pg: svbool_t, op1: svint64_t, op2: svint64_t) -> svint64_t {
    svsubr_s64_m(pg, svsel_s64(pg, op1, svdup_n_s64(0)), op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_s64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_s64_z(pg: svbool_t, op1: svint64_t, op2: i64) -> svint64_t {
    svsubr_s64_z(pg, op1, svdup_n_s64(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_u8_m(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    let op1_s: svint8_t = unsafe { core::mem::transmute(op1) };
    let op2_s: svint8_t = unsafe { core::mem::transmute(op2) };
    
    let res_s: svint8_t = svsubr_s8_m(pg, op1_s, op2_s);
    
    unsafe { core::mem::transmute::<svint8_t, svuint8_t>(res_s) }
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_u8]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_u8_m(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svsubr_u8_m(pg, op1, svdup_n_u8(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_u8_x(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svsubr_u8_m(pg, op1, op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_u8]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_u8_x(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svsubr_u8_x(pg, op1, svdup_n_u8(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_u8_z(pg: svbool_t, op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    svsubr_u8_m(pg, svsel_u8(pg, op1, svdup_n_u8(0)), op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_u8]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_u8_z(pg: svbool_t, op1: svuint8_t, op2: u8) -> svuint8_t {
    svsubr_u8_z(pg, op1, svdup_n_u8(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_u16_m(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    let op1_s: svint16_t = unsafe { core::mem::transmute(op1) };
    let op2_s: svint16_t = unsafe { core::mem::transmute(op2) };
    let res_s: svint16_t = svsubr_s16_m(pg, op1_s, op2_s);
    unsafe { core::mem::transmute::<svint16_t, svuint16_t>(res_s) }
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_u16]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_u16_m(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svsubr_u16_m(pg, op1, svdup_n_u16(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_u16_x(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svsubr_u16_m(pg, op1, op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_u16]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_u16_x(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svsubr_u16_x(pg, op1, svdup_n_u16(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_u16_z(pg: svbool_t, op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    svsubr_u16_m(pg, svsel_u16(pg, op1, svdup_n_u16(0)), op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_u16]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_u16_z(pg: svbool_t, op1: svuint16_t, op2: u16) -> svuint16_t {
    svsubr_u16_z(pg, op1, svdup_n_u16(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_u32_m(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    let op1_s: svint32_t = unsafe { core::mem::transmute(op1) };
    let op2_s: svint32_t = unsafe { core::mem::transmute(op2) };
    let res_s: svint32_t = svsubr_s32_m(pg, op1_s, op2_s);
    unsafe { core::mem::transmute::<svint32_t, svuint32_t>(res_s) }
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_u32]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_u32_m(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svsubr_u32_m(pg, op1, svdup_n_u32(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_u32_x(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svsubr_u32_m(pg, op1, op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_u32]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_u32_x(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svsubr_u32_x(pg, op1, svdup_n_u32(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_u32_z(pg: svbool_t, op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    svsubr_u32_m(pg, svsel_u32(pg, op1, svdup_n_u32(0)), op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_u32]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_u32_z(pg: svbool_t, op1: svuint32_t, op2: u32) -> svuint32_t {
    svsubr_u32_z(pg, op1, svdup_n_u32(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_u64_m(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    let op1_s: svint64_t = unsafe { core::mem::transmute(op1) };
    let op2_s: svint64_t = unsafe { core::mem::transmute(op2) };
    let res_s: svint64_t = svsubr_s64_m(pg, op1_s, op2_s);
    unsafe { core::mem::transmute::<svint64_t, svuint64_t>(res_s) }
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_u64]_m)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_u64_m(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svsubr_u64_m(pg, op1, svdup_n_u64(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_u64_x(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svsubr_u64_m(pg, op1, op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_u64]_x)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_u64_x(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svsubr_u64_x(pg, op1, svdup_n_u64(op2))
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_u64_z(pg: svbool_t, op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    svsubr_u64_m(pg, svsel_u64(pg, op1, svdup_n_u64(0)), op2)
}
#[doc = "Subtract reversed"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svsubr[_n_u64]_z)"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(subr))]
pub fn svsubr_n_u64_z(pg: svbool_t, op1: svuint64_t, op2: u64) -> svuint64_t {
    svsubr_u64_z(pg, op1, svdup_n_u64(op2))
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqadd))]
pub fn svqadd_s8(op1: svint8_t, op2: svint8_t) -> svint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sqadd.x.nxv16i8")]
        fn _svqadd_s8(op1: svint8_t, op2: svint8_t) -> svint8_t;
    }
    unsafe { _svqadd_s8(op1, op2) }
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_n_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqadd))]
pub fn svqadd_n_s8(op1: svint8_t, op2: i8) -> svint8_t {
    svqadd_s8(op1, svdup_n_s8(op2))
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqadd))]
pub fn svqadd_s16(op1: svint16_t, op2: svint16_t) -> svint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sqadd.x.nxv8i16")]
        fn _svqadd_s16(op1: svint16_t, op2: svint16_t) -> svint16_t;
    }
    unsafe { _svqadd_s16(op1, op2) }
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_n_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqadd))]
pub fn svqadd_n_s16(op1: svint16_t, op2: i16) -> svint16_t {
    svqadd_s16(op1, svdup_n_s16(op2))
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqadd))]
pub fn svqadd_s32(op1: svint32_t, op2: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sqadd.x.nxv4i32")]
        fn _svqadd_s32(op1: svint32_t, op2: svint32_t) -> svint32_t;
    }
    unsafe { _svqadd_s32(op1, op2) }
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_n_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqadd))]
pub fn svqadd_n_s32(op1: svint32_t, op2: i32) -> svint32_t {
    svqadd_s32(op1, svdup_n_s32(op2))
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqadd))]
pub fn svqadd_s64(op1: svint64_t, op2: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.sqadd.x.nxv2i64")]
        fn _svqadd_s64(op1: svint64_t, op2: svint64_t) -> svint64_t;
    }
    unsafe { _svqadd_s64(op1, op2) }
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_n_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(sqadd))]
pub fn svqadd_n_s64(op1: svint64_t, op2: i64) -> svint64_t {
    svqadd_s64(op1, svdup_n_s64(op2))
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqadd))]
pub fn svqadd_u8(op1: svuint8_t, op2: svuint8_t) -> svuint8_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.uqadd.x.nxv16i8")]
        fn _svqadd_u8(op1: svuint8_t, op2: svuint8_t) -> svuint8_t;
    }
    unsafe { _svqadd_u8(op1, op2) }
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_n_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqadd))]
pub fn svqadd_n_u8(op1: svuint8_t, op2: u8) -> svuint8_t {
    svqadd_u8(op1, svdup_n_u8(op2))
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqadd))]
pub fn svqadd_u16(op1: svuint16_t, op2: svuint16_t) -> svuint16_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.uqadd.x.nxv8i16")]
        fn _svqadd_u16(op1: svuint16_t, op2: svuint16_t) -> svuint16_t;
    }
    unsafe { _svqadd_u16(op1, op2) }
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_n_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqadd))]
pub fn svqadd_n_u16(op1: svuint16_t, op2: u16) -> svuint16_t {
    svqadd_u16(op1, svdup_n_u16(op2))
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqadd))]
pub fn svqadd_u32(op1: svuint32_t, op2: svuint32_t) -> svuint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.uqadd.x.nxv4i32")]
        fn _svqadd_u32(op1: svuint32_t, op2: svuint32_t) -> svuint32_t;
    }
    unsafe { _svqadd_u32(op1, op2) }
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_n_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqadd))]
pub fn svqadd_n_u32(op1: svuint32_t, op2: u32) -> svuint32_t {
    svqadd_u32(op1, svdup_n_u32(op2))
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqadd))]
pub fn svqadd_u64(op1: svuint64_t, op2: svuint64_t) -> svuint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.uqadd.x.nxv2i64")]
        fn _svqadd_u64(op1: svuint64_t, op2: svuint64_t) -> svuint64_t;
    }
    unsafe { _svqadd_u64(op1, op2) }
}
#[doc = "Saturating add"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svqadd[_n_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(uqadd))]
pub fn svqadd_n_u64(op1: svuint64_t, op2: u64) -> svuint64_t {
    svqadd_u64(op1, svdup_n_u64(op2))
}
#[doc = "Shuffle active elements of vector to the right and fill with zero"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcompact[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(compact))]
pub fn svcompact_f32(pg: svbool_t, op: svfloat32_t) -> svfloat32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.compact.nxv4f32")]
        fn _svcompact_f32(pg: svbool_t, op: svfloat32_t) -> svfloat32_t;
    }
    unsafe { _svcompact_f32(pg, op) }
}
#[doc = "Shuffle active elements of vector to the right and fill with zero"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcompact[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(compact))]
pub fn svcompact_f64(pg: svbool_t, op: svfloat64_t) -> svfloat64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.compact.nxv2f64")]
        fn _svcompact_f64(pg: svbool2_t, op: svfloat64_t) -> svfloat64_t;
    }
    unsafe { _svcompact_f64(simd_cast(pg), op) }
}
#[doc = "Shuffle active elements of vector to the right and fill with zero"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcompact[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(compact))]
pub fn svcompact_s32(pg: svbool_t, op: svint32_t) -> svint32_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.compact.nxv4i32")]
        fn _svcompact_s32(pg: svbool_t, op: svint32_t) -> svint32_t;
    }
    unsafe { _svcompact_s32(pg, op) }
}
#[doc = "Shuffle active elements of vector to the right and fill with zero"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcompact[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(compact))]
pub fn svcompact_s64(pg: svbool_t, op: svint64_t) -> svint64_t {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.compact.nxv2i64")]
        fn _svcompact_s64(pg: svbool2_t, op: svint64_t) -> svint64_t;
    }
    unsafe { _svcompact_s64(simd_cast(pg), op) }
}
#[doc = "Shuffle active elements of vector to the right and fill with zero"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcompact[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(compact))]
pub fn svcompact_u32(pg: svbool_t, op: svuint32_t) -> svuint32_t {
    let op_s: svint32_t = unsafe { core::mem::transmute(op) };
    let res_s: svint32_t = svcompact_s32(pg, op_s);
    unsafe { core::mem::transmute::<svint32_t, svuint32_t>(res_s) }
}
#[doc = "Shuffle active elements of vector to the right and fill with zero"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svcompact[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(compact))]
pub fn svcompact_u64(pg: svbool_t, op: svuint64_t) -> svuint64_t {
    let op_s: svint64_t = unsafe { core::mem::transmute(op) };
    let res_s: svint64_t = svcompact_s64(pg, op_s);
    unsafe { core::mem::transmute::<svint64_t, svuint64_t>(res_s) }
}
#[doc = "Extract element after last"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlasta[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lasta))]
pub fn svlasta_f32(pg: svbool_t, op: svfloat32_t) -> f32 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.lasta.nxv4f32")]
        fn _svlasta_f32(pg: svbool_t, op: svfloat32_t) -> f32;
    }
    unsafe { _svlasta_f32(pg, op) }
}
#[doc = "Extract element after last"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlasta[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lasta))]
pub fn svlasta_f64(pg: svbool_t, op: svfloat64_t) -> f64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.lasta.nxv2f64")]
        fn _svlasta_f64(pg: svbool2_t, op: svfloat64_t) -> f64;
    }
    unsafe { _svlasta_f64(simd_cast(pg), op) }
}
#[doc = "Extract element after last"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlasta[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lasta))]
pub fn svlasta_s8(pg: svbool_t, op: svint8_t) -> i8 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.lasta.nxv16i8")]
        fn _svlasta_s8(pg: svbool_t, op: svint8_t) -> i8;
    }
    unsafe { _svlasta_s8(pg, op) }
}
#[doc = "Extract element after last"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlasta[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lasta))]
pub fn svlasta_s16(pg: svbool_t, op: svint16_t) -> i16 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.lasta.nxv8i16")]
        fn _svlasta_s16(pg: svbool_t, op: svint16_t) -> i16;
    }
    unsafe { _svlasta_s16(pg, op) }
}
#[doc = "Extract element after last"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlasta[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lasta))]
pub fn svlasta_s32(pg: svbool_t, op: svint32_t) -> i32 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.lasta.nxv4i32")]
        fn _svlasta_s32(pg: svbool_t, op: svint32_t) -> i32;
    }
    unsafe { _svlasta_s32(pg, op) }
}
#[doc = "Extract element after last"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlasta[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lasta))]
pub fn svlasta_s64(pg: svbool_t, op: svint64_t) -> i64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.lasta.nxv2i64")]
        fn _svlasta_s64(pg: svbool2_t, op: svint64_t) -> i64;
    }
    unsafe { _svlasta_s64(simd_cast(pg), op) }
}
#[doc = "Extract element after last"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlasta[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lasta))]
pub fn svlasta_u8(pg: svbool_t, op: svuint8_t) -> u8 {
    let op_s: svint8_t = unsafe { core::mem::transmute(op) };
    let res_s: i8 = svlasta_s8(pg, op_s);
    unsafe { core::mem::transmute::<i8, u8>(res_s) }
}
#[doc = "Extract element after last"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlasta[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lasta))]
pub fn svlasta_u16(pg: svbool_t, op: svuint16_t) -> u16 {
    let op_s: svint16_t = unsafe { core::mem::transmute(op) };
    let res_s: i16 = svlasta_s16(pg, op_s);
    unsafe { core::mem::transmute::<i16, u16>(res_s) }
}
#[doc = "Extract element after last"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlasta[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lasta))]
pub fn svlasta_u32(pg: svbool_t, op: svuint32_t) -> u32 {
    let op_s: svint32_t = unsafe { core::mem::transmute(op) };
    let res_s: i32 = svlasta_s32(pg, op_s);
    unsafe { core::mem::transmute::<i32, u32>(res_s) }
}
#[doc = "Extract element after last"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlasta[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lasta))]
pub fn svlasta_u64(pg: svbool_t, op: svuint64_t) -> u64 {
    let op_s: svint64_t = unsafe { core::mem::transmute(op) };
    let res_s: i64 = svlasta_s64(pg, op_s);
    unsafe { core::mem::transmute::<i64, u64>(res_s) }
}
#[doc = "Extract last element"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlastb[_f32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lastb))]
pub fn svlastb_f32(pg: svbool_t, op: svfloat32_t) -> f32 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.lastb.nxv4f32")]
        fn _svlastb_f32(pg: svbool_t, op: svfloat32_t) -> f32;
    }
    unsafe { _svlastb_f32(pg, op) }
}
#[doc = "Extract last element"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlastb[_f64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lastb))]
pub fn svlastb_f64(pg: svbool_t, op: svfloat64_t) -> f64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.lastb.nxv2f64")]
        fn _svlastb_f64(pg: svbool2_t, op: svfloat64_t) -> f64;
    }
    unsafe { _svlastb_f64(simd_cast(pg), op) }
}
#[doc = "Extract last element"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlastb[_s8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lastb))]
pub fn svlastb_s8(pg: svbool_t, op: svint8_t) -> i8 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.lastb.nxv16i8")]
        fn _svlastb_s8(pg: svbool_t, op: svint8_t) -> i8;
    }
    unsafe { _svlastb_s8(pg, op) }
}
#[doc = "Extract last element"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlastb[_s16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lastb))]
pub fn svlastb_s16(pg: svbool_t, op: svint16_t) -> i16 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.lastb.nxv8i16")]
        fn _svlastb_s16(pg: svbool_t, op: svint16_t) -> i16;
    }
    unsafe { _svlastb_s16(pg, op) }
}
#[doc = "Extract last element"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlastb[_s32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lastb))]
pub fn svlastb_s32(pg: svbool_t, op: svint32_t) -> i32 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.lastb.nxv4i32")]
        fn _svlastb_s32(pg: svbool_t, op: svint32_t) -> i32;
    }
    unsafe { _svlastb_s32(pg, op) }
}
#[doc = "Extract last element"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlastb[_s64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lastb))]
pub fn svlastb_s64(pg: svbool_t, op: svint64_t) -> i64 {
    unsafe extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.lastb.nxv2i64")]
        fn _svlastb_s64(pg: svbool2_t, op: svint64_t) -> i64;
    }
    unsafe { _svlastb_s64(simd_cast(pg), op) }
}
#[doc = "Extract last element"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlastb[_u8])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lastb))]
pub fn svlastb_u8(pg: svbool_t, op: svuint8_t) -> u8 {
    let op_s: svint8_t = unsafe { core::mem::transmute(op) };
    let res_s: i8 = svlastb_s8(pg, op_s);
    unsafe { core::mem::transmute::<i8, u8>(res_s) }
}
#[doc = "Extract last element"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlastb[_u16])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lastb))]
pub fn svlastb_u16(pg: svbool_t, op: svuint16_t) -> u16 {
    let op_s: svint16_t = unsafe { core::mem::transmute(op) };
    let res_s: i16 = svlastb_s16(pg, op_s);
    unsafe { core::mem::transmute::<i16, u16>(res_s) }
}
#[doc = "Extract last element"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlastb[_u32])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lastb))]
pub fn svlastb_u32(pg: svbool_t, op: svuint32_t) -> u32 {
    let op_s: svint32_t = unsafe { core::mem::transmute(op) };
    let res_s: i32 = svlastb_s32(pg, op_s);
    unsafe { core::mem::transmute::<i32, u32>(res_s) }
}
#[doc = "Extract last element"]
#[doc = ""]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svlastb[_u64])"]
#[inline]
#[target_feature(enable = "sve")]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
#[cfg_attr(test, assert_instr(lastb))]
pub fn svlastb_u64(pg: svbool_t, op: svuint64_t) -> u64 {
    let op_s: svint64_t = unsafe { core::mem::transmute(op) };
    let res_s: i64 = svlastb_s64(pg, op_s);
    unsafe { core::mem::transmute::<i64, u64>(res_s) }
}
