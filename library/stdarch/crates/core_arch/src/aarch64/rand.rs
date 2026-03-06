//! AArch64 Random Number intrinsics
//!
//! [ACLE documentation](https://arm-software.github.io/acle/main/acle.html#random-number-generation-intrinsics)

unsafe extern "unadjusted" {
    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.rndr"
    )]
    fn rndr_() -> Tuple;

    #[cfg_attr(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        link_name = "llvm.aarch64.rndrrs"
    )]
    fn rndrrs_() -> Tuple;
}

#[repr(C)]
struct Tuple {
    bits: u64,
    status: bool,
}

/// Stores a 64-bit random number into the object pointed to by the argument and returns
/// zero. If the implementation could not generate a random number within a reasonable
/// period of time the object pointed to by the input is set to zero and a non-zero value
/// is returned.
#[inline]
#[target_feature(enable = "rand")]
#[unstable(feature = "stdarch_aarch64_rand", issue = "153514")]
pub unsafe fn __rndr(value: *mut u64) -> i32 {
    let Tuple { bits, status } = rndr_();
    unsafe { *value = bits };
    status as i32
}

/// Reseeds the random number generator. After that stores a 64-bit random number into
/// the object pointed to by the argument and returns zero. If the implementation could
/// not generate a random number within a reasonable period of time the object pointed
/// to by the input is set to zero and a non-zero value is returned.
#[inline]
#[target_feature(enable = "rand")]
#[unstable(feature = "stdarch_aarch64_rand", issue = "153514")]
pub unsafe fn __rndrrs(value: *mut u64) -> i32 {
    let Tuple { bits, status } = rndrrs_();
    unsafe { *value = bits };
    status as i32
}

#[cfg(test)]
mod test {
    use super::*;
    use stdarch_test::assert_instr;

    #[cfg_attr(test, assert_instr(mrs))]
    #[allow(dead_code)]
    #[target_feature(enable = "rand")]
    unsafe fn test_rndr(value: &mut u64) -> i32 {
        __rndr(value)
    }

    #[cfg_attr(test, assert_instr(mrs))]
    #[allow(dead_code)]
    #[target_feature(enable = "rand")]
    unsafe fn test_rndrrs(value: &mut u64) -> i32 {
        __rndrrs(value)
    }
}
