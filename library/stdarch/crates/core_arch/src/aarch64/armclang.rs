//! ARM compiler specific intrinsics
//!
//! # References
//!
//! - [ARM Compiler v 6.10 - armclang Reference Guide][arm_comp_ref]
//!
//! [arm_comp_ref]: https://developer.arm.com/docs/100067/0610

#[cfg(test)]
use stdarch_test::assert_instr;

/// Inserts a breakpoint instruction.
///
/// `VAL` is a compile-time constant integer in range `[0, 65535]`.
///
/// The breakpoint instruction inserted is `BRK` on A64.
#[cfg_attr(test, assert_instr(brk, VAL = 0))]
#[inline(always)]
#[rustc_legacy_const_generics(0)]
pub unsafe fn __breakpoint<const VAL: i32>() {
    static_assert_imm16!(VAL);
    asm!("brk {}", const VAL);
}
