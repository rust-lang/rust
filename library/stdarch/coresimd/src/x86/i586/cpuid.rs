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
pub unsafe fn __cpuid_count(leaf: u32, sub_leaf: u32) -> CpuidResult {
    let mut r = ::core::mem::uninitialized::<CpuidResult>();
    if cfg!(target_arch = "x86") {
        asm!("cpuid"
             : "={eax}"(r.eax), "={ebx}"(r.ebx), "={ecx}"(r.ecx), "={edx}"(r.edx)
             : "{eax}"(leaf), "{ecx}"(sub_leaf)
             : :);
    } else {
        // x86-64 uses %rbx as the base register, so preserve it.
        asm!("cpuid\n"
             : "={eax}"(r.eax), "={ebx}"(r.ebx), "={ecx}"(r.ecx), "={edx}"(r.edx)
             : "{eax}"(leaf), "{ecx}"(sub_leaf)
             : "rbx" :);
    }
    r
}

/// See [`__cpuid_count`](fn.__cpuid_count.html).
#[inline]
#[cfg_attr(test, assert_instr(cpuid))]
pub unsafe fn __cpuid(leaf: u32) -> CpuidResult {
    __cpuid_count(leaf, 0)
}

/// Does the host support the `cpuid` instruction?
#[inline]
pub fn has_cpuid() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        true
    }
    #[cfg(target_arch = "x86")]
    {
        use x86::i386::{__readeflags, __writeeflags};

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
pub unsafe fn __get_cpuid_max(leaf: u32) -> (u32, u32) {
    let CpuidResult { eax, ebx, .. } = __cpuid(leaf);
    (eax, ebx)
}

#[cfg(test)]
mod tests {
    use x86::i586::cpuid;

    #[test]
    fn test_always_has_cpuid() {
        // all currently-tested targets have the instruction
        // FIXME: add targets without `cpuid` to CI
        assert!(cpuid::has_cpuid());
    }

    #[cfg(target_arch = "x86")]
    #[test]
    fn test_has_cpuid() {
        use x86::i386::__readeflags;
        unsafe {
            let before = __readeflags();

            if cpuid::has_cpuid() {
                assert!(before != __readeflags());
            } else {
                assert!(before == __readeflags());
            }
        }
    }
}
