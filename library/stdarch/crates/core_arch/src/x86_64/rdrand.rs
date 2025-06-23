//! RDRAND and RDSEED instructions for returning random numbers from an Intel
//! on-chip hardware random number generator which has been seeded by an
//! on-chip entropy source.

#![allow(clippy::module_name_repetitions)]

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.x86.rdrand.64"]
    fn x86_rdrand64_step() -> (u64, i32);
    #[link_name = "llvm.x86.rdseed.64"]
    fn x86_rdseed64_step() -> (u64, i32);
}

#[cfg(test)]
use stdarch_test::assert_instr;

/// Read a hardware generated 64-bit random value and store the result in val.
/// Returns 1 if a random value was generated, and 0 otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_rdrand64_step)
#[inline]
#[target_feature(enable = "rdrand")]
#[cfg_attr(test, assert_instr(rdrand))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _rdrand64_step(val: &mut u64) -> i32 {
    let (v, flag) = x86_rdrand64_step();
    *val = v;
    flag
}

/// Read a 64-bit NIST SP800-90B and SP800-90C compliant random value and store
/// in val. Return 1 if a random value was generated, and 0 otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_rdseed64_step)
#[inline]
#[target_feature(enable = "rdseed")]
#[cfg_attr(test, assert_instr(rdseed))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _rdseed64_step(val: &mut u64) -> i32 {
    let (v, flag) = x86_rdseed64_step();
    *val = v;
    flag
}
