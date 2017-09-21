//! ARMv7 intrinsics.
//!
//! The reference is [ARMv7-M Architecture Reference Manual (Issue
//! E.b)](http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ddi0403e.b/index.html).

pub use super::v6::*;

/// Count Leading Zeros.
#[inline(always)]
#[cfg_attr(test, assert_instr(clz))]
pub fn _clz_u8(x: u8) -> u8 {
    x.leading_zeros() as u8
}

/// Count Leading Zeros.
#[inline(always)]
#[cfg_attr(test, assert_instr(clz))]
pub fn _clz_u16(x: u16) -> u16 {
    x.leading_zeros() as u16
}

/// Count Leading Zeros.
#[inline(always)]
#[cfg_attr(test, assert_instr(clz))]
pub fn _clz_u32(x: u32) -> u32 {
    x.leading_zeros() as u32
}

#[allow(dead_code)]
extern "C" {
    #[link_name="llvm.bitreverse.i32"]
    fn rbit_u32(i: i32) -> i32;
}

/// Reverse the bit order.
#[inline(always)]
#[cfg_attr(test, assert_instr(rbit))]
pub fn _rbit_u32(x: u32) -> u32 {
    unsafe { rbit_u32(x as i32) as u32 }
}
