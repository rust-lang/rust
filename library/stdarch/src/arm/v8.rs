//! ARMv8 intrinsics.
//!
//! The reference is [ARMv8-A Reference Manual](http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ddi0487a.k_10775/index.html).

pub use super::v7::*;

/// Reverse the order of the bytes.
#[inline(always)]
#[cfg_attr(test, assert_instr(rev))]
pub fn _rev_u64(x: u64) -> u64 {
    x.swap_bytes() as u64
}

/// Count Leading Zeros.
#[inline(always)]
#[cfg_attr(test, assert_instr(clz))]
pub fn _clz_u64(x: u64) -> u64 {
    x.leading_zeros() as u64
}

#[allow(dead_code)]
extern "C" {
    #[link_name="llvm.bitreverse.i64"]
    fn rbit_u64(i: i64) -> i64;
}

/// Reverse the bit order.
#[inline(always)]
#[cfg_attr(test, assert_instr(rbit))]
pub fn _rbit_u64(x: u64) -> u64 {
    unsafe { rbit_u64(x as i64) as u64 }
}

/// Counts the leading most significant bits set.
///
/// When all bits of the operand are set it returns the size of the operand in
/// bits.
#[inline(always)]
// LLVM Bug (should be cls): https://bugs.llvm.org/show_bug.cgi?id=31802
#[cfg_attr(test, assert_instr(clz))] 
pub fn _cls_u32(x: u32) -> u32 {
    u32::leading_zeros(!x) as u32
}

/// Counts the leading most significant bits set.
///
/// When all bits of the operand are set it returns the size of the operand in
/// bits.
#[inline(always)]
// LLVM Bug (should be cls): https://bugs.llvm.org/show_bug.cgi?id=31802
#[cfg_attr(test, assert_instr(clz))] 
pub fn _cls_u64(x: u64) -> u64 {
    u64::leading_zeros(!x) as u64
}
