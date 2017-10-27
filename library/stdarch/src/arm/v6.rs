//! ARMv6 intrinsics.
//!
//! The reference is [ARMv6-M Architecture Reference Manual][armv6m].
//!
//! [armv6m]:
//! http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ddi0419c/index.
//! html

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Reverse the order of the bytes.
#[inline(always)]
#[cfg_attr(test, assert_instr(rev))]
pub unsafe fn _rev_u16(x: u16) -> u16 {
    x.swap_bytes() as u16
}

/// Reverse the order of the bytes.
#[inline(always)]
#[cfg_attr(test, assert_instr(rev))]
pub unsafe fn _rev_u32(x: u32) -> u32 {
    x.swap_bytes() as u32
}

#[cfg(test)]
mod tests {
    use arm::v6;

    #[test]
    fn _rev_u16() {
        unsafe {
            assert_eq!(
                v6::_rev_u16(0b0000_0000_1111_1111_u16),
                0b1111_1111_0000_0000_u16
            );
        }
    }

    #[test]
    fn _rev_u32() {
        unsafe {
            assert_eq!(
                v6::_rev_u32(0b0000_0000_1111_1111_0000_0000_1111_1111_u32),
                0b1111_1111_0000_0000_1111_1111_0000_0000_u32
            );
        }
    }
}
