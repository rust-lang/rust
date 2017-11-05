//! `cpuid` intrinsics

#![cfg_attr(feature = "cargo-clippy", allow(stutter))]

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Result of the `cpuid` instruction.
#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "cargo-clippy", allow(stutter))]
pub struct CpuidResult {
    /// EAX register.
    pub eax: u32,
    /// EBX register.
    pub ebx: u32,
    /// ECX register.
    pub ecx: u32,
    /// EDX register.
    pub edx: u32,
}

/// `cpuid` instruction.
///
/// The [CPUID Wikipedia page][wiki_cpuid] contains how to query which
/// information using the `eax` and `ecx` registers, and the format in
/// which this information is returned in `eax...edx`.
///
/// The `has_cpuid()` intrinsics can be used to query whether the `cpuid`
/// instruction is available.
///
/// The definitive references are:
/// - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2:
///   Instruction Set Reference, A-Z][intel64_ref].
/// - [AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and
///   System Instructions][amd64_ref].
///
/// [wiki_cpuid]: https://en.wikipedia.org/wiki/CPUID
/// [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
/// [amd64_ref]: http://support.amd.com/TechDocs/24594.pdf
#[inline(always)]
#[cfg_attr(test, assert_instr(cpuid))]
pub unsafe fn __cpuid_count(eax: u32, ecx: u32) -> CpuidResult {
    let mut r = ::std::mem::uninitialized::<CpuidResult>();
    asm!("cpuid"
         : "={eax}"(r.eax), "={ebx}"(r.ebx), "={ecx}"(r.ecx), "={edx}"(r.edx)
         : "{eax}"(eax), "{ecx}"(ecx)
         : :);
    r
}

/// `cpuid` instruction.
///
/// See `__cpuid_count`.
#[inline(always)]
#[cfg_attr(test, assert_instr(cpuid))]
pub unsafe fn __cpuid(eax: u32) -> CpuidResult {
    __cpuid_count(eax, 0)
}

/// Does the host support the `cpuid` instruction?
#[inline(always)]
pub fn has_cpuid() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        true
    }
    #[cfg(target_arch = "x86")]
    {
        use super::ia32::{__readeflags, __writeeflags};

        // On `x86` the `cpuid` instruction is not always available.
        // This follows the approach indicated in:
        // http://wiki.osdev.org/CPUID#Checking_CPUID_availability
        unsafe {
            // Read EFLAGS:
            let eflags: u32 = __readeflags();

            // Invert the ID bit in EFLAGS:
            let eflags_mod: u32 = eflags | 0x0020_0000;

            // Store the modified EFLAGS (ID bit may or may not be inverted)
            __writeeflags(eflags_mod);

            // Read EFLAGS again:
            let eflags_after: u32 = __readeflags();

            // Check if the ID bit changed:
            eflags_after != eflags
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_always_has_cpuid() {
        // all currently-tested targets have the instruction
        // FIXME: add targets without `cpuid` to CI
        assert!(has_cpuid());
    }

    #[cfg(target_arch = "x86")]
    #[test]
    fn test_has_cpuid() {
        use vendor::__readeflags;
        unsafe {
            let before = __readeflags();

            if has_cpuid() {
                assert!(before != __readeflags());
            } else {
                assert!(before == __readeflags());
            }
        }
    }

}
