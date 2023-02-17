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
pub unsafe fn _bswap(x: i32) -> i32 {
    x.swap_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bswap() {
        unsafe {
            assert_eq!(_bswap(0x0EADBE0F), 0x0FBEAD0E);
            assert_eq!(_bswap(0x00000000), 0x00000000);
        }
    }
}
