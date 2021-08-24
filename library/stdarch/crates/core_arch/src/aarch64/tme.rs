//! ARM's Transactional Memory Extensions (TME).
//!
//! This CPU feature is available on Aarch64 - A architecture profile.
//! This feature is in the non-neon feature set. TME specific vendor documentation can
//! be found [TME Intrinsics Introduction][tme_intrinsics_intro].
//!
//! The reference is [ACLE Q4 2019][acle_q4_2019_ref].
//!
//! ACLE has a section for TME extensions and state masks for aborts and failure codes.
//! [ARM A64 Architecture Register Datasheet][a_profile_future] also describes possible failure code scenarios.
//!
//! [acle_q4_2019_ref]: https://static.docs.arm.com/101028/0010/ACLE_2019Q4_release-0010.pdf
//! [tme_intrinsics_intro]: https://developer.arm.com/docs/101028/0010/transactional-memory-extension-tme-intrinsics
//! [llvm_aarch64_int]: https://github.com/llvm/llvm-project/commit/a36d31478c182903523e04eb271bbf102bfab2cc#diff-ff24e1c35f4d54f1110ce5d90c709319R626-R646
//! [a_profile_future]: https://static.docs.arm.com/ddi0601/a/SysReg_xml_futureA-2019-04.pdf?_ga=2.116560387.441514988.1590524918-1110153136.1588469296

#[cfg(test)]
use stdarch_test::assert_instr;

extern "unadjusted" {
    #[link_name = "llvm.aarch64.tstart"]
    fn aarch64_tstart() -> u64;
    #[link_name = "llvm.aarch64.tcommit"]
    fn aarch64_tcommit() -> ();
    #[link_name = "llvm.aarch64.tcancel"]
    fn aarch64_tcancel(imm0: u64) -> ();
    #[link_name = "llvm.aarch64.ttest"]
    fn aarch64_ttest() -> u64;
}

/// Transaction successfully started.
pub const _TMSTART_SUCCESS: u64 = 0x00_u64;

/// Extraction mask for failure reason
pub const _TMFAILURE_REASON: u64 = 0x00007FFF_u64;

/// Transaction retry is possible.
pub const _TMFAILURE_RTRY: u64 = 1 << 15;

/// Transaction executed a TCANCEL instruction
pub const _TMFAILURE_CNCL: u64 = 1 << 16;

/// Transaction aborted because a conflict occurred
pub const _TMFAILURE_MEM: u64 = 1 << 17;

/// Fallback error type for any other reason
pub const _TMFAILURE_IMP: u64 = 1 << 18;

/// Transaction aborted because a non-permissible operation was attempted
pub const _TMFAILURE_ERR: u64 = 1 << 19;

/// Transaction aborted due to read or write set limit was exceeded
pub const _TMFAILURE_SIZE: u64 = 1 << 20;

/// Transaction aborted due to transactional nesting level was exceeded
pub const _TMFAILURE_NEST: u64 = 1 << 21;

/// Transaction aborted due to a debug trap.
pub const _TMFAILURE_DBG: u64 = 1 << 22;

/// Transaction failed from interrupt
pub const _TMFAILURE_INT: u64 = 1 << 23;

/// Indicates a TRIVIAL version of TM is available
pub const _TMFAILURE_TRIVIAL: u64 = 1 << 24;

/// Starts a new transaction. When the transaction starts successfully the return value is 0.
/// If the transaction fails, all state modifications are discarded and a cause of the failure
/// is encoded in the return value.
///
/// [ARM TME Intrinsics](https://developer.arm.com/docs/101028/0010/transactional-memory-extension-tme-intrinsics).
#[inline]
#[target_feature(enable = "tme")]
#[cfg_attr(test, assert_instr(tstart))]
pub unsafe fn __tstart() -> u64 {
    aarch64_tstart()
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
#[cfg_attr(test, assert_instr(tcancel, IMM16 = 0x0))]
#[rustc_legacy_const_generics(0)]
pub unsafe fn __tcancel<const IMM16: u64>() {
    static_assert!(IMM16: u64 where IMM16 <= 65535);
    aarch64_tcancel(IMM16);
}

/// Tests if executing inside a transaction. If no transaction is currently executing,
/// the return value is 0. Otherwise, this intrinsic returns the depth of the transaction.
///
/// [ARM TME Intrinsics](https://developer.arm.com/docs/101028/0010/transactional-memory-extension-tme-intrinsics).
#[inline]
#[target_feature(enable = "tme")]
#[cfg_attr(test, assert_instr(ttest))]
pub unsafe fn __ttest() -> u64 {
    aarch64_ttest()
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::aarch64::*;

    const CANCEL_CODE: u64 = (0 | (0x123 & _TMFAILURE_REASON) as u64) as u64;

    #[simd_test(enable = "tme")]
    unsafe fn test_tstart() {
        let mut x = 0;
        for i in 0..10 {
            let code = tme::__tstart();
            if code == _TMSTART_SUCCESS {
                x += 1;
                assert_eq!(x, i + 1);
                break;
            }
            assert_eq!(x, 0);
        }
    }

    #[simd_test(enable = "tme")]
    unsafe fn test_tcommit() {
        let mut x = 0;
        for i in 0..10 {
            let code = tme::__tstart();
            if code == _TMSTART_SUCCESS {
                x += 1;
                assert_eq!(x, i + 1);
                tme::__tcommit();
            }
            assert_eq!(x, i + 1);
        }
    }

    #[simd_test(enable = "tme")]
    unsafe fn test_tcancel() {
        let mut x = 0;

        for i in 0..10 {
            let code = tme::__tstart();
            if code == _TMSTART_SUCCESS {
                x += 1;
                assert_eq!(x, i + 1);
                tme::__tcancel::<CANCEL_CODE>();
                break;
            }
        }

        assert_eq!(x, 0);
    }

    #[simd_test(enable = "tme")]
    unsafe fn test_ttest() {
        for _ in 0..10 {
            let code = tme::__tstart();
            if code == _TMSTART_SUCCESS {
                if tme::__ttest() == 2 {
                    tme::__tcancel::<CANCEL_CODE>();
                    break;
                }
            }
        }
    }
}
