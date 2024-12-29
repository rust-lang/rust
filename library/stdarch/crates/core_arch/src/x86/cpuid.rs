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
/// The highest-supported leaf value is returned by the first tuple argument of
/// [`__get_cpuid_max(0)`](fn.__get_cpuid_max.html). For leaves containing
/// sub-leaves, the second tuple argument returns the highest-supported
/// sub-leaf value.
///
/// The [CPUID Wikipedia page][wiki_cpuid] contains how to query which
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
/// [amd64_ref]: http://support.amd.com/TechDocs/24594.pdf
#[inline]
#[cfg_attr(test, assert_instr(cpuid))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn __cpuid_count(leaf: u32, sub_leaf: u32) -> CpuidResult {
    let eax;
    let ebx;
    let ecx;
    let edx;

    // LLVM sometimes reserves `ebx` for its internal use, we so we need to use
    // a scratch register for it instead.
    #[cfg(target_arch = "x86")]
    {
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
    {
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

/// See [`__cpuid_count`](fn.__cpuid_count.html).
#[inline]
#[cfg_attr(test, assert_instr(cpuid))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn __cpuid(leaf: u32) -> CpuidResult {
    __cpuid_count(leaf, 0)
}

/// Returns the highest-supported `leaf` (`EAX`) and sub-leaf (`ECX`) `cpuid`
/// values.
///
/// If `cpuid` is supported, and `leaf` is zero, then the first tuple argument
/// contains the highest `leaf` value that `cpuid` supports. For `leaf`s
/// containing sub-leafs, the second tuple argument contains the
/// highest-supported sub-leaf value.
///
/// See also [`__cpuid`](fn.__cpuid.html) and
/// [`__cpuid_count`](fn.__cpuid_count.html).
#[inline]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub unsafe fn __get_cpuid_max(leaf: u32) -> (u32, u32) {
    let CpuidResult { eax, ebx, .. } = __cpuid(leaf);
    (eax, ebx)
}
