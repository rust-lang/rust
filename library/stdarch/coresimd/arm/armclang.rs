//! ARM compiler specific intrinsics
//!
//! # References
//!
//! - [ARM Compiler v 6.10 - armclang Reference Guide][arm_comp_ref]
//!
//! [arm_comp_ref]: https://developer.arm.com/docs/100067/0610

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Inserts a breakpoint instruction.
///
/// `val` is a compile-time constant integer in range `[0, 255]`.
///
/// The breakpoint instruction inserted is:
///
/// * `BKPT` when compiling as T32,
/// * `BRK` when compiling as A32 or A64.
///
/// # Safety
///
/// If `val` is out-of-range the behavior is **undefined**.
///
/// # Note
///
/// [ARM's documentation][arm_docs] defines that `__breakpoint` accepts the
/// following values for `val`:
///
/// - `0...65535` when compiling as A32 or A64,
/// - `0...255` when compiling as T32.
///
/// The current implementation only accepts values in range `[0, 255]` - if the
/// value is out-of-range the behavior is **undefined**.
///
/// [arm_docs]: https://developer.arm.com/docs/100067/latest/compiler-specific-intrinsics/__breakpoint-intrinsic
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(bkpt, val = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(brk, val = 0))]
#[inline(always)]
#[rustc_args_required_const(0)]
pub unsafe fn __breakpoint(val: i32) {
    // Ensure that this compiles correctly on non-arm architectures, so libstd
    // doc builds work. The proper macro will shadow this definition below.
    #[allow(unused_macros)]
    macro_rules! call {
        ($e:expr) => {()}
    }

    #[cfg(target_arch = "arm")]
    macro_rules! call {
        ($imm8:expr) => {
            asm!(concat!("BKPT ", stringify!($imm8)) : : : : "volatile")
        }
    }

    #[cfg(target_arch = "aarch64")]
    macro_rules! call {
        ($imm8:expr) => {
            asm!(concat!("BRK ", stringify!($imm8)) : : : : "volatile")
        }
    }

    // We can't `panic!` inside this intrinsic, so we can't really validate the
    // arguments here. If `val` is out-of-range this macro uses `val == 255`:
    constify_imm8!(val, call);
}
