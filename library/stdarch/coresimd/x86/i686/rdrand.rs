//! RDRAND and RDSEED instructions for returning random numbers from an Intel
//! on-chip hardware random number generator which has been seeded by an on-chip
//! entropy source.

extern "platform-intrinsic" {
    fn x86_rdrand16_step() -> (u16, i32);
    fn x86_rdrand32_step() -> (u32, i32);
    fn x86_rdseed16_step() -> (u16, i32);
    fn x86_rdseed32_step() -> (u32, i32);
}

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Read a hardware generated 16-bit random value and store the result in val.
/// Return 1 if a random value was generated, and 0 otherwise.
#[inline]
#[target_feature(enable = "rdrand")]
#[cfg_attr(test, assert_instr(rdrand))]
pub unsafe fn _rdrand16_step(val: &mut u16) -> i32 {
    let (v, flag) = x86_rdrand16_step();
    *val = v;
    flag
}

/// Read a hardware generated 32-bit random value and store the result in val.
/// Return 1 if a random value was generated, and 0 otherwise.
#[inline]
#[target_feature(enable = "rdrand")]
#[cfg_attr(test, assert_instr(rdrand))]
pub unsafe fn _rdrand32_step(val: &mut u32) -> i32 {
    let (v, flag) = x86_rdrand32_step();
    *val = v;
    flag
}

/// Read a 16-bit NIST SP800-90B and SP800-90C compliant random value and store
/// in val. Return 1 if a random value was generated, and 0 otherwise.
#[inline]
#[target_feature(enable = "rdseed")]
#[cfg_attr(test, assert_instr(rdseed))]
pub unsafe fn _rdseed16_step(val: &mut u16) -> i32 {
    let (v, flag) = x86_rdseed16_step();
    *val = v;
    flag
}

/// Read a 32-bit NIST SP800-90B and SP800-90C compliant random value and store
/// in val. Return 1 if a random value was generated, and 0 otherwise.
#[inline]
#[target_feature(enable = "rdseed")]
#[cfg_attr(test, assert_instr(rdseed))]
pub unsafe fn _rdseed32_step(val: &mut u32) -> i32 {
    let (v, flag) = x86_rdseed32_step();
    *val = v;
    flag
}
