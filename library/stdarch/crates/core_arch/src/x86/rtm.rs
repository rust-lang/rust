//! Intel's Restricted Transactional Memory (RTM).
//!
//! This CPU feature is available on Intel Broadwell or later CPUs (and some Haswell).
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref].
//!
//! [Wikipedia][wikipedia_rtm] provides a quick overview of the assembly instructions, and
//! Intel's [programming considerations][intel_consid] details what sorts of instructions within a
//! transaction are likely to cause an abort.
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
//! [wikipedia_rtm]: https://en.wikipedia.org/wiki/Transactional_Synchronization_Extensions#Restricted_Transactional_Memory
//! [intel_consid]: https://software.intel.com/en-us/cpp-compiler-developer-guide-and-reference-intel-transactional-synchronization-extensions-intel-tsx-programming-considerations

#[cfg(test)]
use stdarch_test::assert_instr;

unsafe extern "C" {
    #[link_name = "llvm.x86.xbegin"]
    fn x86_xbegin() -> i32;
    #[link_name = "llvm.x86.xend"]
    fn x86_xend();
    #[link_name = "llvm.x86.xabort"]
    fn x86_xabort(imm8: i8);
    #[link_name = "llvm.x86.xtest"]
    fn x86_xtest() -> i32;
}

/// Transaction successfully started.
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub const _XBEGIN_STARTED: u32 = !0;

/// Transaction explicitly aborted with xabort. The parameter passed to xabort is available with
/// `_xabort_code(status)`.
#[allow(clippy::identity_op)]
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub const _XABORT_EXPLICIT: u32 = 1 << 0;

/// Transaction retry is possible.
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub const _XABORT_RETRY: u32 = 1 << 1;

/// Transaction abort due to a memory conflict with another thread.
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub const _XABORT_CONFLICT: u32 = 1 << 2;

/// Transaction abort due to the transaction using too much memory.
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub const _XABORT_CAPACITY: u32 = 1 << 3;

/// Transaction abort due to a debug trap.
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub const _XABORT_DEBUG: u32 = 1 << 4;

/// Transaction abort in a inner nested transaction.
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub const _XABORT_NESTED: u32 = 1 << 5;

/// Specifies the start of a restricted transactional memory (RTM) code region and returns a value
/// indicating status.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xbegin)
#[inline]
#[target_feature(enable = "rtm")]
#[cfg_attr(test, assert_instr(xbegin))]
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub unsafe fn _xbegin() -> u32 {
    x86_xbegin() as _
}

/// Specifies the end of a restricted transactional memory (RTM) code region.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xend)
#[inline]
#[target_feature(enable = "rtm")]
#[cfg_attr(test, assert_instr(xend))]
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub unsafe fn _xend() {
    x86_xend()
}

/// Forces a restricted transactional memory (RTM) region to abort.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xabort)
#[inline]
#[target_feature(enable = "rtm")]
#[cfg_attr(test, assert_instr(xabort, IMM8 = 0x0))]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub unsafe fn _xabort<const IMM8: u32>() {
    static_assert_uimm_bits!(IMM8, 8);
    x86_xabort(IMM8 as i8)
}

/// Queries whether the processor is executing in a transactional region identified by restricted
/// transactional memory (RTM) or hardware lock elision (HLE).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_xtest)
#[inline]
#[target_feature(enable = "rtm")]
#[cfg_attr(test, assert_instr(xtest))]
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub unsafe fn _xtest() -> u8 {
    x86_xtest() as _
}

/// Retrieves the parameter passed to [`_xabort`] when [`_xbegin`]'s status has the
/// `_XABORT_EXPLICIT` flag set.
#[inline]
#[unstable(feature = "stdarch_x86_rtm", issue = "111138")]
pub const fn _xabort_code(status: u32) -> u32 {
    (status >> 24) & 0xFF
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "rtm")]
    unsafe fn test_xbegin() {
        let mut x = 0;
        for _ in 0..10 {
            let code = _xbegin();
            if code == _XBEGIN_STARTED {
                x += 1;
                _xend();
                assert_eq!(x, 1);
                break;
            }
            assert_eq!(x, 0);
        }
    }

    #[simd_test(enable = "rtm")]
    unsafe fn test_xabort() {
        const ABORT_CODE: u32 = 42;
        // aborting outside a transactional region does nothing
        _xabort::<ABORT_CODE>();

        for _ in 0..10 {
            let mut x = 0;
            let code = rtm::_xbegin();
            if code == _XBEGIN_STARTED {
                x += 1;
                rtm::_xabort::<ABORT_CODE>();
            } else if code & _XABORT_EXPLICIT != 0 {
                let test_abort_code = rtm::_xabort_code(code);
                assert_eq!(test_abort_code, ABORT_CODE);
            }
            assert_eq!(x, 0);
        }
    }

    #[simd_test(enable = "rtm")]
    unsafe fn test_xtest() {
        assert_eq!(_xtest(), 0);

        for _ in 0..10 {
            let code = rtm::_xbegin();
            if code == _XBEGIN_STARTED {
                let in_tx = _xtest();
                rtm::_xend();

                // putting the assert inside the transaction would abort the transaction on fail
                // without any output/panic/etc
                assert_eq!(in_tx, 1);
                break;
            }
        }
    }
}
