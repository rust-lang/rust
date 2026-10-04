//! Byte swap intrinsics.
#![allow(clippy::module_name_repetitions)]

#[cfg(test)]
use stdarch_test::assert_instr;

/// Returns an integer with the reversed byte order of x
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_bswap)
#[inline]
#[cfg_attr(test, assert_instr(bswap))]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[rustc_const_unstable(feature = "stdarch_const_x86", issue = "149298")]
pub const fn _bswap(x: i32) -> i32 {
    x.swap_bytes()
}

#[cfg(test)]
mod tests {
    use crate::core_arch::assert_eq_const as assert_eq;
    use stdarch_test::simd_test;

    use super::*;

    #[simd_test]
    const fn test_bswap() {
        assert_eq!(_bswap(0x0EADBE0F), 0x0FBEAD0E);
        assert_eq!(_bswap(0x00000000), 0x00000000);
    }
}
