//! Byte swap intrinsics.

#![cfg_attr(feature = "cargo-clippy", allow(clippy::stutter))]

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Return an integer with the reversed byte order of x
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_bswap)
#[inline]
#[cfg_attr(test, assert_instr(bswap))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn _bswap(x: i32) -> i32 {
    bswap_i32(x)
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.bswap.i32"]
    fn bswap_i32(x: i32) -> i32;
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
