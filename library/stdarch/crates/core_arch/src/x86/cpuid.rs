//! `cpuid` intrinsics
#![allow(clippy::module_name_repetitions)]

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
/// and
/// `sub_leaf` (`ECX`).
///
/// The highest-supported leaf value is returned by the first tuple argument of
/// [`__get_cpuid_max(0)`](fn.__get_cpuid_max.html). For leaves containung
/// sub-leaves, the second tuple argument returns the highest-supported
/// sub-leaf
/// value.
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
/// [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
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
            "movl %ebx, {0}",
            "cpuid",
            "xchgl %ebx, {0}",
            lateout(reg) ebx,
            inlateout("eax") leaf => eax,
            inlateout("ecx") sub_leaf => ecx,
            lateout("edx") edx,
            options(nostack, preserves_flags, att_syntax),
        );
    }
    #[cfg(target_arch = "x86_64")]
    {
        asm!(
            "movq %rbx, {0:r}",
            "cpuid",
            "xchgq %rbx, {0:r}",
            lateout(reg) ebx,
            inlateout("eax") leaf => eax,
            inlateout("ecx") sub_leaf => ecx,
            lateout("edx") edx,
            options(nostack, preserves_flags, att_syntax),
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

/// Does the host support the `cpuid` instruction?
#[inline]
pub fn has_cpuid() -> bool {
    #[cfg(target_env = "sgx")]
    {
        false
    }
    #[cfg(all(not(target_env = "sgx"), target_arch = "x86_64"))]
    {
        true
    }
    #[cfg(all(not(target_env = "sgx"), target_arch = "x86"))]
    {
        // Optimization for i586 and i686 Rust targets which SSE enabled
        // and support cpuid:
        #[cfg(target_feature = "sse")]
        {
            true
        }

        // If SSE is not enabled, detect whether cpuid is available:
        #[cfg(not(target_feature = "sse"))]
        unsafe {
            // On `x86` the `cpuid` instruction is not always available.
            // This follows the approach indicated in:
            // http://wiki.osdev.org/CPUID#Checking_CPUID_availability
            // https://software.intel.com/en-us/articles/using-cpuid-to-detect-the-presence-of-sse-41-and-sse-42-instruction-sets/
            // which detects whether `cpuid` is available by checking whether
            // the 21st bit of the EFLAGS register is modifiable or not.
            // If it is, then `cpuid` is available.
            let result: u32;
            asm!(
                // Read eflags and save a copy of it
                "pushfd",
                "pop {result}",
                "mov {result}, {saved_flags}",
                // Flip 21st bit of the flags
                "xor $0x200000, {result}",
                // Load the modified flags and read them back.
                // Bit 21 can only be modified if cpuid is available.
                "push {result}",
                "popfd",
                "pushfd",
                "pop {result}",
                // Use xor to find out whether bit 21 has changed
                "xor {saved_flags}, {result}",
                result = out(reg) result,
                saved_flags = out(reg) _,
                options(nomem, att_syntax),
            );
            // There is a race between popfd (A) and pushfd (B)
            // where other bits beyond 21st may have been modified due to
            // interrupts, a debugger stepping through the asm, etc.
            //
            // Therefore, explicitly check whether the 21st bit
            // was modified or not.
            //
            // If the result is zero, the cpuid bit was not modified.
            // If the result is `0x200000` (non-zero), then the cpuid
            // was correctly modified and the CPU supports the cpuid
            // instruction:
            (result & 0x200000) != 0
        }
    }
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

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;

    #[test]
    fn test_always_has_cpuid() {
        // all currently-tested targets have the instruction
        // FIXME: add targets without `cpuid` to CI
        assert!(cpuid::has_cpuid());
    }

    #[test]
    fn test_has_cpuid_idempotent() {
        assert_eq!(cpuid::has_cpuid(), cpuid::has_cpuid());
    }
}
