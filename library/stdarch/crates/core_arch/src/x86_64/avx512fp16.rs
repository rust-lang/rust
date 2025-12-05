use crate::core_arch::x86::*;
#[cfg(test)]
use stdarch_test::assert_instr;

/// Convert the signed 64-bit integer b to a half-precision (16-bit) floating-point element, store the
/// result in the lower element of dst, and copy the upper 3 packed elements from a to the upper elements
/// of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvti64_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vcvtsi2sh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub fn _mm_cvti64_sh(a: __m128h, b: i64) -> __m128h {
    unsafe { vcvtsi642sh(a, b, _MM_FROUND_CUR_DIRECTION) }
}

/// Convert the signed 64-bit integer b to a half-precision (16-bit) floating-point element, store the
/// result in the lower element of dst, and copy the upper 3 packed elements from a to the upper elements
/// of dst.
///
/// Rounding is done according to the rounding parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvt_roundi64_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vcvtsi2sh, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub fn _mm_cvt_roundi64_sh<const ROUNDING: i32>(a: __m128h, b: i64) -> __m128h {
    unsafe {
        static_assert_rounding!(ROUNDING);
        vcvtsi642sh(a, b, ROUNDING)
    }
}

/// Convert the unsigned 64-bit integer b to a half-precision (16-bit) floating-point element, store the
/// result in the lower element of dst, and copy the upper 1 packed elements from a to the upper elements
/// of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtu64_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vcvtusi2sh))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub fn _mm_cvtu64_sh(a: __m128h, b: u64) -> __m128h {
    unsafe { vcvtusi642sh(a, b, _MM_FROUND_CUR_DIRECTION) }
}

/// Convert the unsigned 64-bit integer b to a half-precision (16-bit) floating-point element, store the
/// result in the lower element of dst, and copy the upper 1 packed elements from a to the upper elements
/// of dst.
///
/// Rounding is done according to the rounding parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvt_roundu64_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vcvtusi2sh, ROUNDING = 8))]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub fn _mm_cvt_roundu64_sh<const ROUNDING: i32>(a: __m128h, b: u64) -> __m128h {
    unsafe {
        static_assert_rounding!(ROUNDING);
        vcvtusi642sh(a, b, ROUNDING)
    }
}

/// Convert the lower half-precision (16-bit) floating-point element in a to a 64-bit integer, and store
/// the result in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtsh_i64)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vcvtsh2si))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub fn _mm_cvtsh_i64(a: __m128h) -> i64 {
    unsafe { vcvtsh2si64(a, _MM_FROUND_CUR_DIRECTION) }
}

/// Convert the lower half-precision (16-bit) floating-point element in a to a 64-bit integer, and store
/// the result in dst.
///
/// Rounding is done according to the rounding parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvt_roundsh_i64)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vcvtsh2si, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub fn _mm_cvt_roundsh_i64<const ROUNDING: i32>(a: __m128h) -> i64 {
    unsafe {
        static_assert_rounding!(ROUNDING);
        vcvtsh2si64(a, ROUNDING)
    }
}

/// Convert the lower half-precision (16-bit) floating-point element in a to a 64-bit unsigned integer, and store
/// the result in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtsh_u64)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vcvtsh2usi))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub fn _mm_cvtsh_u64(a: __m128h) -> u64 {
    unsafe { vcvtsh2usi64(a, _MM_FROUND_CUR_DIRECTION) }
}

/// Convert the lower half-precision (16-bit) floating-point element in a to a 64-bit unsigned integer, and store
/// the result in dst.
///
/// Rounding is done according to the rounding parameter, which can be one of:
///
/// * [`_MM_FROUND_TO_NEAREST_INT`] | [`_MM_FROUND_NO_EXC`] : round to nearest and suppress exceptions
/// * [`_MM_FROUND_TO_NEG_INF`] | [`_MM_FROUND_NO_EXC`] : round down and suppress exceptions
/// * [`_MM_FROUND_TO_POS_INF`] | [`_MM_FROUND_NO_EXC`] : round up and suppress exceptions
/// * [`_MM_FROUND_TO_ZERO`] | [`_MM_FROUND_NO_EXC`] : truncate and suppress exceptions
/// * [`_MM_FROUND_CUR_DIRECTION`] : use `MXCSR.RC` - see [`_MM_SET_ROUNDING_MODE`]
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvt_roundsh_u64)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vcvtsh2usi, ROUNDING = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub fn _mm_cvt_roundsh_u64<const ROUNDING: i32>(a: __m128h) -> u64 {
    unsafe {
        static_assert_rounding!(ROUNDING);
        vcvtsh2usi64(a, ROUNDING)
    }
}

/// Convert the lower half-precision (16-bit) floating-point element in a to a 64-bit integer with truncation,
/// and store the result in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvttsh_i64)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vcvttsh2si))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub fn _mm_cvttsh_i64(a: __m128h) -> i64 {
    unsafe { vcvttsh2si64(a, _MM_FROUND_CUR_DIRECTION) }
}

/// Convert the lower half-precision (16-bit) floating-point element in a to a 64-bit integer with truncation,
/// and store the result in dst.
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtt_roundsh_i64)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vcvttsh2si, SAE = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub fn _mm_cvtt_roundsh_i64<const SAE: i32>(a: __m128h) -> i64 {
    unsafe {
        static_assert_sae!(SAE);
        vcvttsh2si64(a, SAE)
    }
}

/// Convert the lower half-precision (16-bit) floating-point element in a to a 64-bit unsigned integer with truncation,
/// and store the result in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvttsh_u64)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vcvttsh2usi))]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub fn _mm_cvttsh_u64(a: __m128h) -> u64 {
    unsafe { vcvttsh2usi64(a, _MM_FROUND_CUR_DIRECTION) }
}

/// Convert the lower half-precision (16-bit) floating-point element in a to a 64-bit unsigned integer with truncation,
/// and store the result in dst.
///
/// Exceptions can be suppressed by passing _MM_FROUND_NO_EXC in the sae parameter.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtt_roundsh_u64)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[cfg_attr(test, assert_instr(vcvttsh2usi, SAE = 8))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub fn _mm_cvtt_roundsh_u64<const SAE: i32>(a: __m128h) -> u64 {
    unsafe {
        static_assert_sae!(SAE);
        vcvttsh2usi64(a, SAE)
    }
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.x86.avx512fp16.vcvtsi642sh"]
    fn vcvtsi642sh(a: __m128h, b: i64, rounding: i32) -> __m128h;
    #[link_name = "llvm.x86.avx512fp16.vcvtusi642sh"]
    fn vcvtusi642sh(a: __m128h, b: u64, rounding: i32) -> __m128h;
    #[link_name = "llvm.x86.avx512fp16.vcvtsh2si64"]
    fn vcvtsh2si64(a: __m128h, rounding: i32) -> i64;
    #[link_name = "llvm.x86.avx512fp16.vcvtsh2usi64"]
    fn vcvtsh2usi64(a: __m128h, rounding: i32) -> u64;
    #[link_name = "llvm.x86.avx512fp16.vcvttsh2si64"]
    fn vcvttsh2si64(a: __m128h, sae: i32) -> i64;
    #[link_name = "llvm.x86.avx512fp16.vcvttsh2usi64"]
    fn vcvttsh2usi64(a: __m128h, sae: i32) -> u64;
}

#[cfg(test)]
mod tests {
    use crate::core_arch::{x86::*, x86_64::*};
    use stdarch_test::simd_test;

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_cvti64_sh() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let r = _mm_cvti64_sh(a, 10);
        let e = _mm_setr_ph(10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_cvt_roundi64_sh() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let r = _mm_cvt_roundi64_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a, 10);
        let e = _mm_setr_ph(10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_cvtu64_sh() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let r = _mm_cvtu64_sh(a, 10);
        let e = _mm_setr_ph(10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_cvt_roundu64_sh() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let r = _mm_cvt_roundu64_sh::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a, 10);
        let e = _mm_setr_ph(10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_cvtsh_i64() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let r = _mm_cvtsh_i64(a);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_cvt_roundsh_i64() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let r = _mm_cvt_roundsh_i64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_cvtsh_u64() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let r = _mm_cvtsh_u64(a);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_cvt_roundsh_u64() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let r = _mm_cvt_roundsh_u64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_cvttsh_i64() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let r = _mm_cvttsh_i64(a);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_cvtt_roundsh_i64() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let r = _mm_cvtt_roundsh_i64::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(a);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_cvttsh_u64() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let r = _mm_cvttsh_u64(a);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_cvtt_roundsh_u64() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let r = _mm_cvtt_roundsh_u64::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 1);
    }
}
