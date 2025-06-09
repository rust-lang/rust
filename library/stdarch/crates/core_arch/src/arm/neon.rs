use crate::core_arch::arm_shared::neon::*;

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.arm.neon.vbsl.v8i8"]
    fn vbsl_s8_(a: int8x8_t, b: int8x8_t, c: int8x8_t) -> int8x8_t;
    #[link_name = "llvm.arm.neon.vbsl.v16i8"]
    fn vbslq_s8_(a: int8x16_t, b: int8x16_t, c: int8x16_t) -> int8x16_t;
}

#[doc = "Shift Left and Insert (immediate)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsli_n_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7,aes")]
#[unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsli.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsli_n_p64<const N: i32>(a: poly64x1_t, b: poly64x1_t) -> poly64x1_t {
    static_assert!(0 <= N && N <= 63);
    transmute(vshiftins_v1i64(
        transmute(a),
        transmute(b),
        int64x1_t::splat(N as i64),
    ))
}

#[doc = "Shift Left and Insert (immediate)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsliq_n_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[cfg(target_endian = "little")]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7,aes")]
#[unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsli.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_p64<const N: i32>(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    static_assert!(0 <= N && N <= 63);
    transmute(vshiftins_v2i64(
        transmute(a),
        transmute(b),
        int64x2_t::splat(N as i64),
    ))
}

#[doc = "Shift Left and Insert (immediate)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsliq_n_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[cfg(target_endian = "big")]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7,aes")]
#[unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsli.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsliq_n_p64<const N: i32>(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    static_assert!(0 <= N && N <= 63);
    let a: poly64x2_t = simd_shuffle!(a, a, [0, 1]);
    let b: poly64x2_t = simd_shuffle!(b, b, [0, 1]);
    let ret_val: poly64x2_t = transmute(vshiftins_v2i64(
        transmute(a),
        transmute(b),
        int64x2_t::splat(N as i64),
    ));
    simd_shuffle!(ret_val, ret_val, [0, 1])
}

#[doc = "Shift Right and Insert (immediate)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsri_n_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7,aes")]
#[unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsri.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsri_n_p64<const N: i32>(a: poly64x1_t, b: poly64x1_t) -> poly64x1_t {
    static_assert!(1 <= N && N <= 64);
    transmute(vshiftins_v1i64(
        transmute(a),
        transmute(b),
        int64x1_t::splat(-N as i64),
    ))
}

#[doc = "Shift Right and Insert (immediate)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsriq_n_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[cfg(target_endian = "little")]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7,aes")]
#[unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsri.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_p64<const N: i32>(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    static_assert!(1 <= N && N <= 64);
    transmute(vshiftins_v2i64(
        transmute(a),
        transmute(b),
        int64x2_t::splat(-N as i64),
    ))
}

#[doc = "Shift Right and Insert (immediate)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsriq_n_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[cfg(target_endian = "big")]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7,aes")]
#[unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsri.64", N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsriq_n_p64<const N: i32>(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    static_assert!(1 <= N && N <= 64);
    let a: poly64x2_t = simd_shuffle!(a, a, [0, 1]);
    let b: poly64x2_t = simd_shuffle!(b, b, [0, 1]);
    let ret_val: poly64x2_t = transmute(vshiftins_v2i64(
        transmute(a),
        transmute(b),
        int64x2_t::splat(-N as i64),
    ));
    simd_shuffle!(ret_val, ret_val, [0, 1])
}
