//! ARM intrinsics.
//!
//! The reference for NEON is [ARM's NEON Intrinsics Reference][arm_ref]. The
//! [ARM's NEON Intrinsics Online Database][arm_dat] is also useful.
//!
//! [arm_ref]: http://infocenter.arm.com/help/topic/com.arm.doc.ihi0073a/IHI0073A_arm_neon_intrinsics_ref.pdf
//! [arm_dat]: https://developer.arm.com/technologies/neon/intrinsics

mod armclang;
pub use self::armclang::*;

mod v6;
pub use self::v6::*;

// Supported arches: 6, 7-M. See Section 10.1 of ACLE (e.g. SSAT)
#[cfg(any(target_feature = "v6", doc))]
mod sat;

#[cfg(any(target_feature = "v6", doc))]
pub use self::sat::*;

// Supported arches: 5TE, 7E-M. See Section 10.1 of ACLE (e.g. QADD)
// We also include the A profile even though DSP is deprecated on that profile as of ACLE 2.0 (see
// section 5.4.7)
// Here we workaround the difference between LLVM's +dsp and ACLE's __ARM_FEATURE_DSP by gating on
// '+v5te' rather than on '+dsp'
#[cfg(any(
    // >= v5TE but excludes v7-M
    all(target_feature = "v5te", not(target_feature = "mclass")),
    // v7E-M
    all(target_feature = "mclass", target_feature = "dsp"),
    doc,
))]
pub mod dsp;

#[cfg(any(
    // >= v5TE but excludes v7-M
    all(target_feature = "v5te", not(target_feature = "mclass")),
    // v7E-M
    all(target_feature = "mclass", target_feature = "dsp"),
    doc,
))]
pub use self::dsp::*;

// Deprecated in ACLE 2.0 for the A profile but fully supported on the M and R profiles, says
// Section 5.4.9 of ACLE. We'll expose these for the A profile even if deprecated
#[cfg(any(
    // v7-A, v7-R
    all(target_feature = "v6", not(target_feature = "mclass")),
    // v7E-M
    all(target_feature = "mclass", target_feature = "dsp"),
    doc,
))]
mod simd32;

#[cfg(any(
    // v7-A, v7-R
    all(target_feature = "v6", not(target_feature = "mclass")),
    // v7E-M
    all(target_feature = "mclass", target_feature = "dsp"),
    doc,
))]
pub use self::simd32::*;

#[cfg(any(target_feature = "v7", doc))]
mod v7;
#[cfg(any(target_feature = "v7", doc))]
pub use self::v7::*;

mod ex;
pub use self::ex::*;

pub use crate::core_arch::arm_shared::*;

#[cfg(test)]
use stdarch_test::assert_instr;

#[cfg(any(target_feature = "v7", doc))]
pub(crate) mod neon;
#[cfg(any(target_feature = "v7", doc))]
pub use neon::*;

/// Generates the trap instruction `UDF`
#[cfg(target_arch = "arm")]
#[cfg_attr(test, assert_instr(udf))]
#[inline]
pub unsafe fn udf() -> ! {
    crate::intrinsics::abort()
}

/// Generates a DBG instruction.
///
/// This provides a hint to debugging and related systems. The argument must be
/// a constant integer from 0 to 15 inclusive. See implementation documentation
/// for the effect (if any) of this instruction and the meaning of the
/// argument. This is available only when compliling for AArch32.
// Section 10.1 of ACLE says that the supported arches are: 7, 7-M
// "The DBG hint instruction is added in ARMv7. It is UNDEFINED in the ARMv6 base architecture, and
// executes as a NOP instruction in ARMv6K and ARMv6T2." - ARM Architecture Reference Manual ARMv7-A
// and ARMv7-R edition (ARM DDI 0406C.c) sections D12.4.1 "ARM instruction set support" and D12.4.2
// "Thumb instruction set support"
#[cfg(any(target_feature = "v7", doc))]
#[inline(always)]
#[rustc_legacy_const_generics(0)]
pub unsafe fn __dbg<const IMM4: i32>() {
    static_assert_imm4!(IMM4);
    dbg(IMM4);
}

extern "unadjusted" {
    #[link_name = "llvm.arm.dbg"]
    fn dbg(_: i32);
}
