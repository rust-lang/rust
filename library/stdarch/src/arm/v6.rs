//! ARMv6 intrinsics.
//!
//! The reference is [ARMv6-M Architecture Reference
//! Manual](http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ddi0419c/index.html).

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Reverse the order of the bytes.
#[inline(always)]
#[cfg_attr(test, assert_instr(rev))]
pub fn _rev_u16(x: u16) -> u16 {
    x.swap_bytes() as u16
}

/// Reverse the order of the bytes.
#[inline(always)]
#[cfg_attr(test, assert_instr(rev))]
pub fn _rev_u32(x: u32) -> u32 {
    x.swap_bytes() as u32
}
