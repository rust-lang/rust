//! ARMv7 intrinsics.
//!
//! The reference is [ARMv7-M Architecture Reference Manual (Issue
//! E.b)][armv7m].
//!
//! [armv7m]:
//! http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ddi0403e.
//! b/index.html

pub use super::v6::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Count Leading Zeros.
#[inline]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
// FIXME: https://github.com/rust-lang-nursery/stdsimd/issues/382
// #[cfg_attr(all(test, target_arch = "arm"), assert_instr(clz))]
pub unsafe fn _clz_u8(x: u8) -> u8 {
    x.leading_zeros() as u8
}

/// Count Leading Zeros.
#[inline]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
// FIXME: https://github.com/rust-lang-nursery/stdsimd/issues/382
// #[cfg_attr(all(test, target_arch = "arm"), assert_instr(clz))]
pub unsafe fn _clz_u16(x: u16) -> u16 {
    x.leading_zeros() as u16
}

/// Count Leading Zeros.
#[inline]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
// FIXME: https://github.com/rust-lang-nursery/stdsimd/issues/382
// #[cfg_attr(all(test, target_arch = "arm"), assert_instr(clz))]
pub unsafe fn _clz_u32(x: u32) -> u32 {
    x.leading_zeros() as u32
}

/// Reverse the bit order.
#[inline]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn _rbit_u32(x: u32) -> u32 {
    use intrinsics::bitreverse;
    bitreverse(x)
}

#[cfg(test)]
mod tests {
    use core_arch::arm::v7;

    #[test]
    fn _clz_u8() {
        unsafe {
            assert_eq!(v7::_clz_u8(0b0000_1010u8), 4u8);
        }
    }

    #[test]
    fn _clz_u16() {
        unsafe {
            assert_eq!(v7::_clz_u16(0b0000_1010u16), 12u16);
        }
    }

    #[test]
    fn _clz_u32() {
        unsafe {
            assert_eq!(v7::_clz_u32(0b0000_1010u32), 28u32);
        }
    }

    #[test]
    #[cfg(dont_compile_me)] // FIXME need to add `v7` upstream in rustc
    fn _rbit_u32() {
        unsafe {
            assert_eq!(
                v7::_rbit_u32(0b0000_1010u32),
                0b0101_0000_0000_0000_0000_0000_0000_0000u32
            );
        }
    }
}
