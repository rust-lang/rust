#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.x86.addcarry.64"]
    fn llvm_addcarry_u64(a: u8, b: u64, c: u64) -> (u8, u64);
    #[link_name = "llvm.x86.addcarryx.u64"]
    fn llvm_addcarryx_u64(a: u8, b: u64, c: u64, d: *mut u64) -> u8;
    #[link_name = "llvm.x86.subborrow.64"]
    fn llvm_subborrow_u64(a: u8, b: u64, c: u64) -> (u8, u64);
}

/// Adds unsigned 64-bit integers `a` and `b` with unsigned 8-bit carry-in `c_in`
/// (carry or overflow flag), and store the unsigned 64-bit result in `out`, and the carry-out
/// is returned (carry or overflow flag).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_addcarry_u64)
#[inline]
#[cfg_attr(test, assert_instr(adc))]
#[stable(feature = "simd_x86_adx", since = "1.33.0")]
pub unsafe fn _addcarry_u64(c_in: u8, a: u64, b: u64, out: &mut u64) -> u8 {
    let (a, b) = llvm_addcarry_u64(c_in, a, b);
    *out = b;
    a
}

/// Adds unsigned 64-bit integers `a` and `b` with unsigned 8-bit carry-in `c_in`
/// (carry or overflow flag), and store the unsigned 64-bit result in `out`, and
/// the carry-out is returned (carry or overflow flag).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_addcarryx_u64)
#[inline]
#[target_feature(enable = "adx")]
#[cfg_attr(test, assert_instr(adc))]
#[stable(feature = "simd_x86_adx", since = "1.33.0")]
pub unsafe fn _addcarryx_u64(c_in: u8, a: u64, b: u64, out: &mut u64) -> u8 {
    llvm_addcarryx_u64(c_in, a, b, out as *mut _)
}

/// Adds unsigned 64-bit integers `a` and `b` with unsigned 8-bit carry-in `c_in`.
/// (carry or overflow flag), and store the unsigned 64-bit result in `out`, and
/// the carry-out is returned (carry or overflow flag).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_subborrow_u64)
#[inline]
#[cfg_attr(test, assert_instr(sbb))]
#[stable(feature = "simd_x86_adx", since = "1.33.0")]
pub unsafe fn _subborrow_u64(c_in: u8, a: u64, b: u64, out: &mut u64) -> u8 {
    let (a, b) = llvm_subborrow_u64(c_in, a, b);
    *out = b;
    a
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::x86_64::*;

    #[test]
    fn test_addcarry_u64() {
        unsafe {
            let a = u64::MAX;
            let mut out = 0;

            let r = _addcarry_u64(0, a, 1, &mut out);
            assert_eq!(r, 1);
            assert_eq!(out, 0);

            let r = _addcarry_u64(0, a, 0, &mut out);
            assert_eq!(r, 0);
            assert_eq!(out, a);

            let r = _addcarry_u64(1, a, 1, &mut out);
            assert_eq!(r, 1);
            assert_eq!(out, 1);

            let r = _addcarry_u64(1, a, 0, &mut out);
            assert_eq!(r, 1);
            assert_eq!(out, 0);

            let r = _addcarry_u64(0, 3, 4, &mut out);
            assert_eq!(r, 0);
            assert_eq!(out, 7);

            let r = _addcarry_u64(1, 3, 4, &mut out);
            assert_eq!(r, 0);
            assert_eq!(out, 8);
        }
    }

    #[simd_test(enable = "adx")]
    unsafe fn test_addcarryx_u64() {
        let a = u64::MAX;
        let mut out = 0;

        let r = _addcarry_u64(0, a, 1, &mut out);
        assert_eq!(r, 1);
        assert_eq!(out, 0);

        let r = _addcarry_u64(0, a, 0, &mut out);
        assert_eq!(r, 0);
        assert_eq!(out, a);

        let r = _addcarry_u64(1, a, 1, &mut out);
        assert_eq!(r, 1);
        assert_eq!(out, 1);

        let r = _addcarry_u64(1, a, 0, &mut out);
        assert_eq!(r, 1);
        assert_eq!(out, 0);

        let r = _addcarry_u64(0, 3, 4, &mut out);
        assert_eq!(r, 0);
        assert_eq!(out, 7);

        let r = _addcarry_u64(1, 3, 4, &mut out);
        assert_eq!(r, 0);
        assert_eq!(out, 8);
    }

    #[test]
    fn test_subborrow_u64() {
        unsafe {
            let a = u64::MAX;
            let mut out = 0;

            let r = _subborrow_u64(0, 0, 1, &mut out);
            assert_eq!(r, 1);
            assert_eq!(out, a);

            let r = _subborrow_u64(0, 0, 0, &mut out);
            assert_eq!(r, 0);
            assert_eq!(out, 0);

            let r = _subborrow_u64(1, 0, 1, &mut out);
            assert_eq!(r, 1);
            assert_eq!(out, a - 1);

            let r = _subborrow_u64(1, 0, 0, &mut out);
            assert_eq!(r, 1);
            assert_eq!(out, a);

            let r = _subborrow_u64(0, 7, 3, &mut out);
            assert_eq!(r, 0);
            assert_eq!(out, 4);

            let r = _subborrow_u64(1, 7, 3, &mut out);
            assert_eq!(r, 0);
            assert_eq!(out, 3);
        }
    }
}
