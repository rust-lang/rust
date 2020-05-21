//! ARM's Transactional Memory Extensions (TME).
//!
//! This CPU feature is available on Aarch64 – ARMv8-A arch. onwards.
//! This feature is in the non-neon feature set. TME specific vendor documentation can
//! be found [TME Intrinsics Introduction][tme_intrinsics_intro].
//!
//! The reference is [ACLE Q4 2019][acle_q4_2019_ref].
//!
//! ACLE has a section for TME extensions and state masks for aborts and failure codes.
//! In addition to that [LLVM Aarch64 Intrinsics][llvm_aarch64_int] are
//! self explanatory for what needs to be exported.
//!
//! [acle_q4_2019_ref]: https://static.docs.arm.com/101028/0010/ACLE_2019Q4_release-0010.pdf
//! [tme_intrinsics_intro]: https://developer.arm.com/docs/101028/0010/transactional-memory-extension-tme-intrinsics
//! [llvm_aarch64_int]: https://github.com/llvm/llvm-project/commit/a36d31478c182903523e04eb271bbf102bfab2cc#diff-ff24e1c35f4d54f1110ce5d90c709319R626-R646

#[cfg(test)]
use stdarch_test::assert_instr;

extern "C" {
    #[link_name = "llvm.aarch64.tstart"]
    fn aarch64_tstart() -> i32;
    #[link_name = "llvm.aarch64.tcommit"]
    fn aarch64_tcommit() -> ();
    #[link_name = "llvm.aarch64.tcancel"]
    fn aarch64_tcancel(imm0: i64) -> ();
    #[link_name = "llvm.aarch64.ttest"]
    fn aarch64_ttest() -> i32;
}

/// Transaction successfully started.
pub const _TMSTART_SUCCESS: u32 = 0x00_u32;

/// Extraction mask for failure reason
pub const _TMFAILURE_REASON: u32 = 0x00007FFF_u32;

/// Transaction retry is possible.
pub const _TMFAILURE_RTRY: u32 = 1 << 15;

/// Transaction cancelled.
pub const _TMFAILURE_CNCL: u32 = 1 << 16;

/// Transaction cancelled due to high memory usage.
pub const _TMFAILURE_MEM: u32 = 1 << 17;

///
pub const _TMFAILURE_IMP: u32 = 1 << 18;

///
pub const _TMFAILURE_ERR: u32 = 1 << 19;

///
pub const _TMFAILURE_SIZE: u32 = 1 << 20;

/// Transaction abort in a inner nested transaction.
pub const _TMFAILURE_NEST: u32 = 1 << 21;

/// Transaction abort due to a debug trap.
pub const _TMFAILURE_DBG: u32 = 1 << 22;

///
pub const _TMFAILURE_INT: u32 = 1 << 23;

///
pub const _TMFAILURE_TRIVIAL: u32 = 1 << 24;


/// Starts a new transaction. When the transaction starts successfully the return value is 0.
/// If the transaction fails, all state modifications are discarded and a cause of the failure
/// is encoded in the return value.
///
/// [ARM TME Intrinsics](https://developer.arm.com/docs/101028/0010/transactional-memory-extension-tme-intrinsics).
#[inline]
#[target_feature(enable = "tme")]
#[cfg_attr(test, assert_instr(tstart))]
pub unsafe fn __tstart() -> u32 {
    aarch64_tstart() as _
}

/// Commits the current transaction. For a nested transaction, the only effect is that the
/// transactional nesting depth is decreased. For an outer transaction, the state modifications
/// performed transactionally are committed to the architectural state.
///
/// [ARM TME Intrinsics](https://developer.arm.com/docs/101028/0010/transactional-memory-extension-tme-intrinsics).
#[inline]
#[target_feature(enable = "tme")]
#[cfg_attr(test, assert_instr(tcommit))]
pub unsafe fn __tcommit() {
    aarch64_tcommit()
}

/// Cancels the current transaction and discards all state modifications that were performed transactionally.
///
/// [ARM TME Intrinsics](https://developer.arm.com/docs/101028/0010/transactional-memory-extension-tme-intrinsics).
#[inline]
#[target_feature(enable = "tme")]
#[cfg_attr(test, assert_instr(tcancel, imm0 = 0x0))]
#[rustc_args_required_const(0)]
pub unsafe fn __tcancel(imm0: u32) {
    macro_rules! call {
        ($imm0:expr) => {
            aarch64_tcancel($imm0)
        };
    }
    constify_imm8!(imm0, call)
}

/// Tests if executing inside a transaction. If no transaction is currently executing,
/// the return value is 0. Otherwise, this intrinsic returns the depth of the transaction.
///
/// [ARM TME Intrinsics](https://developer.arm.com/docs/101028/0010/transactional-memory-extension-tme-intrinsics).
#[inline]
#[target_feature(enable = "tme")]
#[cfg_attr(test, assert_instr(ttest))]
pub unsafe fn __ttest() -> u32 {
    aarch64_ttest() as _
}

/// Encodes cancellation reason, which is the parameter passed to [`__tcancel`]
/// Takes cancellation reason flags and retry-ability.
#[inline]
pub const fn _tcancel_code(reason: u32, retryable: bool) -> u32 {
    (retryable as i32) << 15 | (reason & _TMFAILURE_REASON)
}
