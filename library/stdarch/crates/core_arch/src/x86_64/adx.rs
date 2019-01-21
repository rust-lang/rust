#[cfg(test)]
use stdsimd_test::assert_instr;

#[allow(improper_ctypes)]
extern "unadjusted" {
    #[link_name = "llvm.x86.addcarry.u64"]
    fn llvm_addcarry_u64(a: u8, b: u64, c: u64) -> (u8, u64);
    #[link_name = "llvm.x86.subborrow.u64"]
    fn llvm_subborrow_u64(a: u8, b: u64, c: u64) -> (u8, u64);
}

/// Add unsigned 64-bit integers a and b with unsigned 8-bit carry-in c_in
/// (carry flag), and store the unsigned 64-bit result in out, and the carry-out
/// is returned (carry or overflow flag).
#[inline]
#[cfg_attr(test, assert_instr(adc))]
#[stable(feature = "simd_x86_adx", since = "1.33.0")]
pub unsafe fn _addcarry_u64(c_in: u8, a: u64, b: u64, out: &mut u64) -> u8 {
    let (a, b) = llvm_addcarry_u64(c_in, a, b);
    *out = b;
    a
}

/// Add unsigned 64-bit integers a and b with unsigned 8-bit carry-in c_in
/// (carry or overflow flag), and store the unsigned 64-bit result in out, and
/// the carry-out is returned (carry or overflow flag).
#[inline]
#[target_feature(enable = "adx")]
#[cfg_attr(test, assert_instr(adc))]
#[stable(feature = "simd_x86_adx", since = "1.33.0")]
#[cfg(not(stage0))]
pub unsafe fn _addcarryx_u64(c_in: u8, a: u64, b: u64, out: &mut u64) -> u8 {
    _addcarry_u64(c_in, a, b, out)
}

/// Add unsigned 64-bit integers a and b with unsigned 8-bit carry-in c_in
/// (carry or overflow flag), and store the unsigned 64-bit result in out, and
/// the carry-out is returned (carry or overflow flag).
#[inline]
#[cfg_attr(test, assert_instr(sbb))]
#[stable(feature = "simd_x86_adx", since = "1.33.0")]
pub unsafe fn _subborrow_u64(c_in: u8, a: u64, b: u64, out: &mut u64) -> u8 {
    let (a, b) = llvm_subborrow_u64(c_in, a, b);
    *out = b;
    a
}
