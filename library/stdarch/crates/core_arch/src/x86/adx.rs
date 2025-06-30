#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.x86.addcarry.32"]
    fn llvm_addcarry_u32(a: u8, b: u32, c: u32) -> (u8, u32);
    #[link_name = "llvm.x86.addcarryx.u32"]
    fn llvm_addcarryx_u32(a: u8, b: u32, c: u32, d: *mut u32) -> u8;
    #[link_name = "llvm.x86.subborrow.32"]
    fn llvm_subborrow_u32(a: u8, b: u32, c: u32) -> (u8, u32);
}

/// Adds unsigned 32-bit integers `a` and `b` with unsigned 8-bit carry-in `c_in`
/// (carry or overflow flag), and store the unsigned 32-bit result in `out`, and the carry-out
/// is returned (carry or overflow flag).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_addcarry_u32)
#[inline]
#[cfg_attr(test, assert_instr(adc))]
#[stable(feature = "simd_x86_adx", since = "1.33.0")]
pub unsafe fn _addcarry_u32(c_in: u8, a: u32, b: u32, out: &mut u32) -> u8 {
    let (a, b) = llvm_addcarry_u32(c_in, a, b);
    *out = b;
    a
}

/// Adds unsigned 32-bit integers `a` and `b` with unsigned 8-bit carry-in `c_in`
/// (carry or overflow flag), and store the unsigned 32-bit result in `out`, and
/// the carry-out is returned (carry or overflow flag).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_addcarryx_u32)
#[inline]
#[target_feature(enable = "adx")]
#[cfg_attr(test, assert_instr(adc))]
#[stable(feature = "simd_x86_adx", since = "1.33.0")]
pub unsafe fn _addcarryx_u32(c_in: u8, a: u32, b: u32, out: &mut u32) -> u8 {
    llvm_addcarryx_u32(c_in, a, b, out as *mut _)
}

/// Adds unsigned 32-bit integers `a` and `b` with unsigned 8-bit carry-in `c_in`
/// (carry or overflow flag), and store the unsigned 32-bit result in `out`, and
/// the carry-out is returned (carry or overflow flag).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_subborrow_u32)
#[inline]
#[cfg_attr(test, assert_instr(sbb))]
#[stable(feature = "simd_x86_adx", since = "1.33.0")]
pub unsafe fn _subborrow_u32(c_in: u8, a: u32, b: u32, out: &mut u32) -> u8 {
    let (a, b) = llvm_subborrow_u32(c_in, a, b);
    *out = b;
    a
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[test]
    fn test_addcarry_u32() {
        unsafe {
            let a = u32::MAX;
            let mut out = 0;

            let r = _addcarry_u32(0, a, 1, &mut out);
            assert_eq!(r, 1);
            assert_eq!(out, 0);

            let r = _addcarry_u32(0, a, 0, &mut out);
            assert_eq!(r, 0);
            assert_eq!(out, a);

            let r = _addcarry_u32(1, a, 1, &mut out);
            assert_eq!(r, 1);
            assert_eq!(out, 1);

            let r = _addcarry_u32(1, a, 0, &mut out);
            assert_eq!(r, 1);
            assert_eq!(out, 0);

            let r = _addcarry_u32(0, 3, 4, &mut out);
            assert_eq!(r, 0);
            assert_eq!(out, 7);

            let r = _addcarry_u32(1, 3, 4, &mut out);
            assert_eq!(r, 0);
            assert_eq!(out, 8);
        }
    }

    #[simd_test(enable = "adx")]
    unsafe fn test_addcarryx_u32() {
        let a = u32::MAX;
        let mut out = 0;

        let r = _addcarryx_u32(0, a, 1, &mut out);
        assert_eq!(r, 1);
        assert_eq!(out, 0);

        let r = _addcarryx_u32(0, a, 0, &mut out);
        assert_eq!(r, 0);
        assert_eq!(out, a);

        let r = _addcarryx_u32(1, a, 1, &mut out);
        assert_eq!(r, 1);
        assert_eq!(out, 1);

        let r = _addcarryx_u32(1, a, 0, &mut out);
        assert_eq!(r, 1);
        assert_eq!(out, 0);

        let r = _addcarryx_u32(0, 3, 4, &mut out);
        assert_eq!(r, 0);
        assert_eq!(out, 7);

        let r = _addcarryx_u32(1, 3, 4, &mut out);
        assert_eq!(r, 0);
        assert_eq!(out, 8);
    }

    #[simd_test(enable = "adx")]
    unsafe fn test_addcarryx_u32_2() {
        unsafe fn add_1_2_3() -> u32 {
            let mut out = 0;
            _addcarryx_u32(1, 2, 3, &mut out);
            out
        }
        assert_eq!(6, add_1_2_3());
    }

    #[test]
    fn test_subborrow_u32() {
        unsafe {
            let a = u32::MAX;
            let mut out = 0;

            let r = _subborrow_u32(0, 0, 1, &mut out);
            assert_eq!(r, 1);
            assert_eq!(out, a);

            let r = _subborrow_u32(0, 0, 0, &mut out);
            assert_eq!(r, 0);
            assert_eq!(out, 0);

            let r = _subborrow_u32(1, 0, 1, &mut out);
            assert_eq!(r, 1);
            assert_eq!(out, a - 1);

            let r = _subborrow_u32(1, 0, 0, &mut out);
            assert_eq!(r, 1);
            assert_eq!(out, a);

            let r = _subborrow_u32(0, 7, 3, &mut out);
            assert_eq!(r, 0);
            assert_eq!(out, 4);

            let r = _subborrow_u32(1, 7, 3, &mut out);
            assert_eq!(r, 0);
            assert_eq!(out, 3);
        }
    }
}
