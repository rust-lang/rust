//! `cpuid` intrinsics
#![allow(clippy::module_name_repetitions)]

use crate::arch::asm;
#[cfg(test)]
use stdarch_test::assert_instr;

/// Result of the `cpuid` instruction.
#[allow(clippy::missing_inline_in_public_items)]
// ^^ the derived impl of Debug for CpuidResult is not #[inline] and that's OK.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub struct CpuidResult {
    /// EAX register.
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub eax: u32,
    /// EBX register.
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub ebx: u32,
    /// ECX register.
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub ecx: u32,
    /// EDX register.
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub edx: u32,
}

/// Returns the result of the `cpuid` instruction for a given `leaf` (`EAX`)
/// and `sub_leaf` (`ECX`).
///
/// There are two types of information leaves - basic leaves (with `leaf < 0x8000000`)
/// and extended leaves (with `leaf >= 0x80000000`). The highest supported basic and
/// extended leaves can be obtained by calling CPUID with `0` and `0x80000000`,
/// respectively, and reading the value in the `EAX` register. If the leaf supports
/// more than one sub-leaf, then the procedure of obtaining the highest supported
/// sub-leaf, as well as the behavior if a invalid sub-leaf value is passed, depends
/// on the specific leaf.
///
/// If the `leaf` value is higher than the maximum supported basic or extended leaf
/// for the processor, this returns the information for the highest supported basic
/// information leaf (with the passed `sub_leaf` value). If the `leaf` value is less
/// than or equal to the highest basic or extended leaf value, but the leaf is not
/// supported on the processor, all zeros are returned.
///
/// The [CPUID Wikipedia page][wiki_cpuid] contains information on how to query which
/// information using the `EAX` and `ECX` registers, and the interpretation of
/// the results returned in `EAX`, `EBX`, `ECX`, and `EDX`.
///
/// The references are:
/// - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2:
///   Instruction Set Reference, A-Z][intel64_ref].
/// - [AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and
///   System Instructions][amd64_ref].
///
/// [wiki_cpuid]: https://en.wikipedia.org/wiki/CPUID
/// [intel64_ref]: https://cdrdv2-public.intel.com/671110/325383-sdm-vol-2abcd.pdf
/// [amd64_ref]: https://docs.amd.com/v/u/en-US/24594_3.37
#[inline]
#[cfg_attr(test, assert_instr(cpuid))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn __cpuid_count(leaf: u32, sub_leaf: u32) -> CpuidResult {
    if cfg!(target_env = "sgx") {
        panic!("`__cpuid` cannot be used in SGX");
    }

    let eax;
    let ebx;
    let ecx;
    let edx;

    // LLVM sometimes reserves `ebx` for its internal use, we so we need to use
    // a scratch register for it instead.
    #[cfg(target_arch = "x86")]
    unsafe {
        asm!(
            "mov {0}, ebx",
            "cpuid",
            "xchg {0}, ebx",
            out(reg) ebx,
            inout("eax") leaf => eax,
            inout("ecx") sub_leaf => ecx,
            out("edx") edx,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        asm!(
            "mov {0:r}, rbx",
            "cpuid",
            "xchg {0:r}, rbx",
            out(reg) ebx,
            inout("eax") leaf => eax,
            inout("ecx") sub_leaf => ecx,
            out("edx") edx,
            options(nostack, preserves_flags),
        );
    }
    CpuidResult { eax, ebx, ecx, edx }
}

/// Calls CPUID with the provided `leaf` value, with `sub_leaf` set to 0.
/// See [`__cpuid_count`](fn.__cpuid_count.html).
#[inline]
#[cfg_attr(test, assert_instr(cpuid))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn __cpuid(leaf: u32) -> CpuidResult {
    __cpuid_count(leaf, 0)
}

/// Returns the EAX and EBX register after calling CPUID with the provided `leaf`,
/// with `sub_leaf` set to 0.
///
/// If `leaf` if 0 or `0x80000000`, the first tuple argument contains the maximum
/// supported basic or extended leaf, respectively.
///
/// See also [`__cpuid`](fn.__cpuid.html) and
/// [`__cpuid_count`](fn.__cpuid_count.html).
#[inline]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn __get_cpuid_max(leaf: u32) -> (u32, u32) {
    let CpuidResult { eax, ebx, .. } = __cpuid(leaf);
    (eax, ebx)
}
