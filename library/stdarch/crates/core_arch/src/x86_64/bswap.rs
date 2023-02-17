//! Byte swap intrinsics.

#![allow(clippy::module_name_repetitions)]

#[cfg(test)]
use stdarch_test::assert_instr;

/// Returns an integer with the reversed byte order of x
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_bswap64)
#[inline]
#[cfg_attr(test, assert_instr(bswap))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _bswap64(x: i64) -> i64 {
    x.swap_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bswap64() {
        unsafe {
            assert_eq!(_bswap64(0x0EADBEEFFADECA0E), 0x0ECADEFAEFBEAD0E);
            assert_eq!(_bswap64(0x0000000000000000), 0x0000000000000000);
        }
    }
}
