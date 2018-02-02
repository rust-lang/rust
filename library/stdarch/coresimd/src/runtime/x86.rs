//! This module implements minimal run-time feature detection for x86.
//!
//! The features are detected using the `detect_features` function below.
//! This function uses the CPUID instruction to read the feature flags from the
//! CPU and encodes them in an `usize` where each bit position represents
//! whether a feature is available (bit is set) or unavaiable (bit is cleared).
//!
//! The enum `__Feature` is used to map bit positions to feature names, and the
//! the `__unstable_detect_feature!` macro is used to map string literals (e.g.
//! "avx") to these bit positions (e.g. `__Feature::avx`).
//!
//!
//! The run-time feature detection is performed by the
//! `__unstable_detect_feature(__Feature) -> bool` function. On its first call,
//! this functions queries the CPU for the available features and stores them
//! in a global `AtomicUsize` variable. The query is performed by just checking
//! whether the feature bit in this global variable is set or cleared.

use core::mem;

use super::{bit, cache};

/// This macro maps the string-literal feature names to values of the
/// `__Feature` enum at compile-time. The feature names used are the same as
/// those of rustc `target_feature` and `cfg_target_feature` features.
///
/// PLESE: do not use this, it is an implementation detail subjected to change.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_export]
#[doc(hidden)]
macro_rules! __unstable_detect_feature {
    ("aes", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::aes{})  };
    ("tsc", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::tsc{})  };
    ("mmx", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::mmx{})  };
    ("sse", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::sse{})  };
    ("sse2", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::sse2{})
    };
    ("sse3", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::sse3{})
    };
    ("ssse3", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::ssse3{})
    };
    ("sse4.1", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::sse4_1{})
    };
    ("sse4.2", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::sse4_2{})
    };
    ("sse4a", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::sse4a{})
    };
    ("avx", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::avx{})
    };
    ("avx2", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::avx2{})
    };
    ("avx512f", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::avx512f{})
    };
    ("avx512cd", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::avx512cd{})
    };
    ("avx512er", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::avx512er{})
    };
    ("avx512pf", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::avx512pf{})
    };
    ("avx512bw", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::avx512bw{})
    };
    ("avx512dq", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::avx512dq{})
    };
    ("avx512vl", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::avx512vl{})
    };
    ("avx512ifma", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::avx512_ifma{})
    };
    ("avx512vbmi", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::avx512_vbmi{})
    };
    ("avx512vpopcntdq", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::avx512_vpopcntdq{})
    };
    ("fma", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::fma{})
    };
    ("bmi", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::bmi{})
    };
    ("bmi2", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::bmi2{})
    };
    ("abm", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::abm{})
    };
    ("lzcnt", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::abm{})
    };
    ("tbm", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::tbm{})
    };
    ("popcnt", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::popcnt{})
    };
    ("fxsr", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::fxsr{})
    };
    ("xsave", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::xsave{})
    };
    ("xsaveopt", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::xsaveopt{})
    };
    ("xsaves", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::xsaves{})
    };
    ("xsavec", $unstable_detect_feature:path) => {
        $unstable_detect_feature(
            $crate::__vendor_runtime::__Feature::xsavec{})
    };
    ($t:tt, $unstable_detect_feature:path) => {
        compile_error!(concat!("unknown target feature: ", $t))
    };
}

/// X86 CPU Feature enum. Each variant denotes a position in a bitset for a
/// particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum __Feature {
    /// AES (Advanced Encryption Standard New Instructions AES-NI)
    aes,
    /// TSC (Time Stamp Counter)
    tsc,
    /// MMX
    mmx,
    /// SSE (Streaming SIMD Extensions)
    sse,
    /// SSE2 (Streaming SIMD Extensions 2)
    sse2,
    /// SSE3 (Streaming SIMD Extensions 3)
    sse3,
    /// SSSE3 (Supplemental Streaming SIMD Extensions 3)
    ssse3,
    /// SSE4.1 (Streaming SIMD Extensions 4.1)
    sse4_1,
    /// SSE4.2 (Streaming SIMD Extensions 4.2)
    sse4_2,
    /// SSE4a (Streaming SIMD Extensions 4a)
    sse4a,
    /// AVX (Advanced Vector Extensions)
    avx,
    /// AVX2 (Advanced Vector Extensions 2)
    avx2,
    /// AVX-512 F (Foundation)
    avx512f,
    /// AVX-512 CD (Conflict Detection Instructions)
    avx512cd,
    /// AVX-512 ER (Exponential and Reciprocal Instructions)
    avx512er,
    /// AVX-512 PF (Prefetch Instructions)
    avx512pf,
    /// AVX-512 BW (Byte and Word Instructions)
    avx512bw,
    /// AVX-512 DQ (Doubleword and Quadword)
    avx512dq,
    /// AVX-512 VL (Vector Length Extensions)
    avx512vl,
    /// AVX-512 IFMA (Integer Fused Multiply Add)
    avx512_ifma,
    /// AVX-512 VBMI (Vector Byte Manipulation Instructions)
    avx512_vbmi,
    /// AVX-512 VPOPCNTDQ (Vector Population Count Doubleword and
    /// Quadword)
    avx512_vpopcntdq,
    /// FMA (Fused Multiply Add)
    fma,
    /// BMI1 (Bit Manipulation Instructions 1)
    bmi,
    /// BMI1 (Bit Manipulation Instructions 2)
    bmi2,
    /// ABM (Advanced Bit Manipulation) on AMD / LZCNT (Leading Zero
    /// Count) on Intel
    abm,
    /// TBM (Trailing Bit Manipulation)
    tbm,
    /// POPCNT (Population Count)
    popcnt,
    /// FXSR (Floating-point context fast save and restor)
    fxsr,
    /// XSAVE (Save Processor Extended States)
    xsave,
    /// XSAVEOPT (Save Processor Extended States Optimized)
    xsaveopt,
    /// XSAVES (Save Processor Extended States Supervisor)
    xsaves,
    /// XSAVEC (Save Processor Extended States Compacted)
    xsavec,
    #[doc(hidden)]
    __NonExhaustive,
}

/// Run-time feature detection on x86 works by using the CPUID instruction.
///
/// The [CPUID Wikipedia page][wiki_cpuid] contains
/// all the information about which flags to set to query which values, and in
/// which registers these are reported.
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
pub fn detect_features() -> cache::Initializer {
    use vendor::{__cpuid, __cpuid_count, has_cpuid, CpuidResult};
    use vendor::_xgetbv;
    let mut value = cache::Initializer::default();

    // If the x86 CPU does not support the CPUID instruction then it is too
    // old to support any of the currently-detectable features.
    if !has_cpuid() {
        return value;
    }

    // Calling `__cpuid`/`__cpuid_count` from here on is safe because the CPU
    // has `cpuid` support.

    // 0. EAX = 0: Basic Information:
    // - EAX returns the "Highest Function Parameter", that is, the maximum
    // leaf value for subsequent calls of `cpuinfo` in range [0,
    // 0x8000_0000]. - The vendor ID is stored in 12 u8 ascii chars,
    // returned in EBX, EDX, and   ECX (in that order):
    let (max_basic_leaf, vendor_id) = unsafe {
        let CpuidResult {
            eax: max_basic_leaf,
            ebx,
            ecx,
            edx,
        } = __cpuid(0);
        let vendor_id: [[u8; 4]; 3] = [
            mem::transmute(ebx),
            mem::transmute(edx),
            mem::transmute(ecx),
        ];
        let vendor_id: [u8; 12] = mem::transmute(vendor_id);
        (max_basic_leaf, vendor_id)
    };

    if max_basic_leaf < 1 {
        // Earlier Intel 486, CPUID not implemented
        return value;
    }

    // EAX = 1, ECX = 0: Queries "Processor Info and Feature Bits";
    // Contains information about most x86 features.
    let CpuidResult {
        ecx: proc_info_ecx,
        edx: proc_info_edx,
        ..
    } = unsafe { __cpuid(0x0000_0001_u32) };

    // EAX = 7, ECX = 0: Queries "Extended Features";
    // Contains information about bmi,bmi2, and avx2 support.
    let (extended_features_ebx, extended_features_ecx) = if max_basic_leaf >= 7
    {
        let CpuidResult { ebx, ecx, .. } = unsafe { __cpuid(0x0000_0007_u32) };
        (ebx, ecx)
    } else {
        (0, 0) // CPUID does not support "Extended Features"
    };

    // EAX = 0x8000_0000, ECX = 0: Get Highest Extended Function Supported
    // - EAX returns the max leaf value for extended information, that is,
    // `cpuid` calls in range [0x8000_0000; u32::MAX]:
    let CpuidResult {
        eax: extended_max_basic_leaf,
        ..
    } = unsafe { __cpuid(0x8000_0000_u32) };

    // EAX = 0x8000_0001, ECX=0: Queries "Extended Processor Info and Feature
    // Bits"
    let extended_proc_info_ecx = if extended_max_basic_leaf >= 1 {
        let CpuidResult { ecx, .. } = unsafe { __cpuid(0x8000_0001_u32) };
        ecx
    } else {
        0
    };

    {
        // borrows value till the end of this scope:
        let mut enable = |r, rb, f| {
            if bit::test(r as usize, rb) {
                value.set(f as u32);
            }
        };

        enable(proc_info_ecx, 0, __Feature::sse3);
        enable(proc_info_ecx, 9, __Feature::ssse3);
        enable(proc_info_ecx, 12, __Feature::fma);
        enable(proc_info_ecx, 19, __Feature::sse4_1);
        enable(proc_info_ecx, 20, __Feature::sse4_2);
        enable(proc_info_ecx, 23, __Feature::popcnt);
        enable(proc_info_ecx, 25, __Feature::aes);
        enable(proc_info_edx, 4, __Feature::tsc);
        enable(proc_info_edx, 23, __Feature::mmx);
        enable(proc_info_edx, 24, __Feature::fxsr);
        enable(proc_info_edx, 25, __Feature::sse);
        enable(proc_info_edx, 26, __Feature::sse2);

        enable(extended_features_ebx, 3, __Feature::bmi);
        enable(extended_features_ebx, 8, __Feature::bmi2);

        // `XSAVE` and `AVX` support:
        let cpu_xsave = bit::test(proc_info_ecx as usize, 26);
        if cpu_xsave {
            // 0. Here the CPU supports `XSAVE`.

            // 1. Detect `OSXSAVE`, that is, whether the OS is AVX enabled and
            // supports saving the state of the AVX/AVX2 vector registers on
            // context-switches, see:
            //
            // - [intel: is avx enabled?][is_avx_enabled],
            // - [mozilla: sse.cpp][mozilla_sse_cpp].
            //
            // [is_avx_enabled]: https://software.intel.com/en-us/blogs/2011/04/14/is-avx-enabled
            // [mozilla_sse_cpp]: https://hg.mozilla.org/mozilla-central/file/64bab5cbb9b6/mozglue/build/SSE.cpp#l190
            let cpu_osxsave = bit::test(proc_info_ecx as usize, 27);

            // 2. The OS must have signaled the CPU that it supports saving and
            // restoring the SSE and AVX registers by setting `XCR0.SSE[1]` and
            // `XCR0.AVX[2]` to `1`.
            //
            // This is safe because the CPU supports `xsave`
            let xcr0 = unsafe { _xgetbv(0) };
            let os_avx_support = xcr0 & 6 == 6;
            let os_avx512_support = xcr0 & 224 == 224;

            // Only if the OS and the CPU support saving/restoring the AVX
            // registers we enable `xsave` support:
            if cpu_osxsave && os_avx_support {
                // See "13.3 ENABLING THE XSAVE FEATURE SET AND XSAVE-ENABLED
                // FEATURES" in the "Intel® 64 and IA-32 Architectures Software
                // Developer’s Manual, Volume 1: Basic Architecture":
                //
                // "Software enables the XSAVE feature set by setting
                // CR4.OSXSAVE[bit 18] to 1 (e.g., with the MOV to CR4
                // instruction). If this bit is 0, execution of any of XGETBV,
                // XRSTOR, XRSTORS, XSAVE, XSAVEC, XSAVEOPT, XSAVES, and XSETBV
                // causes an invalid-opcode exception (#UD)"
                //
                enable(proc_info_ecx, 26, __Feature::xsave);

                // For `xsaveopt`, `xsavec`, and `xsaves` we need to query:
                // Processor Extended State Enumeration Sub-leaf (EAX = 0DH,
                // ECX = 1):
                if max_basic_leaf >= 0xd {
                    let CpuidResult {
                        eax: proc_extended_state1_eax,
                        ..
                    } = unsafe { __cpuid_count(0xd_u32, 1) };
                    enable(proc_extended_state1_eax, 0, __Feature::xsaveopt);
                    enable(proc_extended_state1_eax, 1, __Feature::xsavec);
                    enable(proc_extended_state1_eax, 3, __Feature::xsaves);
                }

                // And AVX/AVX2:
                enable(proc_info_ecx, 28, __Feature::avx);
                enable(extended_features_ebx, 5, __Feature::avx2);

                // For AVX-512 the OS also needs to support saving/restoring
                // the extended state, only then we enable AVX-512 support:
                if os_avx512_support {
                    enable(extended_features_ebx, 16, __Feature::avx512f);
                    enable(extended_features_ebx, 17, __Feature::avx512dq);
                    enable(extended_features_ebx, 21, __Feature::avx512_ifma);
                    enable(extended_features_ebx, 26, __Feature::avx512pf);
                    enable(extended_features_ebx, 27, __Feature::avx512er);
                    enable(extended_features_ebx, 28, __Feature::avx512cd);
                    enable(extended_features_ebx, 30, __Feature::avx512bw);
                    enable(extended_features_ebx, 31, __Feature::avx512vl);
                    enable(extended_features_ecx, 1, __Feature::avx512_vbmi);
                    enable(
                        extended_features_ecx,
                        14,
                        __Feature::avx512_vpopcntdq,
                    );
                }
            }
        }

        // This detects ABM on AMD CPUs and LZCNT on Intel CPUs.
        // On intel CPUs with popcnt, lzcnt implements the
        // "missing part" of ABM, so we map both to the same
        // internal feature.
        //
        // The `cfg_feature_enabled!("lzcnt")` macro then
        // internally maps to __Feature::abm.
        enable(extended_proc_info_ecx, 5, __Feature::abm);
        if vendor_id == *b"AuthenticAMD" {
            // These features are only available on AMD CPUs:
            enable(extended_proc_info_ecx, 6, __Feature::sse4a);
            enable(extended_proc_info_ecx, 21, __Feature::tbm);
        }
    }

    value
}

#[cfg(test)]
mod tests {
    extern crate cupid;

    #[test]
    fn dump() {
        println!("aes: {:?}", cfg_feature_enabled!("aes"));
        println!("tsc: {:?}", cfg_feature_enabled!("tsc"));
        println!("sse: {:?}", cfg_feature_enabled!("sse"));
        println!("sse2: {:?}", cfg_feature_enabled!("sse2"));
        println!("sse3: {:?}", cfg_feature_enabled!("sse3"));
        println!("ssse3: {:?}", cfg_feature_enabled!("ssse3"));
        println!("sse4.1: {:?}", cfg_feature_enabled!("sse4.1"));
        println!("sse4.2: {:?}", cfg_feature_enabled!("sse4.2"));
        println!("sse4a: {:?}", cfg_feature_enabled!("sse4a"));
        println!("avx: {:?}", cfg_feature_enabled!("avx"));
        println!("avx2: {:?}", cfg_feature_enabled!("avx2"));
        println!("avx512f {:?}", cfg_feature_enabled!("avx512f"));
        println!("avx512cd {:?}", cfg_feature_enabled!("avx512cd"));
        println!("avx512er {:?}", cfg_feature_enabled!("avx512er"));
        println!("avx512pf {:?}", cfg_feature_enabled!("avx512pf"));
        println!("avx512bw {:?}", cfg_feature_enabled!("avx512bw"));
        println!("avx512dq {:?}", cfg_feature_enabled!("avx512dq"));
        println!("avx512vl {:?}", cfg_feature_enabled!("avx512vl"));
        println!("avx512_ifma {:?}", cfg_feature_enabled!("avx512ifma"));
        println!("avx512_vbmi {:?}", cfg_feature_enabled!("avx512vbmi"));
        println!(
            "avx512_vpopcntdq {:?}",
            cfg_feature_enabled!("avx512vpopcntdq")
        );
        println!("fma: {:?}", cfg_feature_enabled!("fma"));
        println!("abm: {:?}", cfg_feature_enabled!("abm"));
        println!("bmi: {:?}", cfg_feature_enabled!("bmi"));
        println!("bmi2: {:?}", cfg_feature_enabled!("bmi2"));
        println!("tbm: {:?}", cfg_feature_enabled!("tbm"));
        println!("popcnt: {:?}", cfg_feature_enabled!("popcnt"));
        println!("lzcnt: {:?}", cfg_feature_enabled!("lzcnt"));
        println!("fxsr: {:?}", cfg_feature_enabled!("fxsr"));
        println!("xsave: {:?}", cfg_feature_enabled!("xsave"));
        println!("xsaveopt: {:?}", cfg_feature_enabled!("xsaveopt"));
        println!("xsaves: {:?}", cfg_feature_enabled!("xsaves"));
        println!("xsavec: {:?}", cfg_feature_enabled!("xsavec"));
    }

    #[test]
    fn compare_with_cupid() {
        let information = cupid::master().unwrap();
        assert_eq!(cfg_feature_enabled!("aes"), information.aesni());
        assert_eq!(cfg_feature_enabled!("tsc"), information.tsc());
        assert_eq!(cfg_feature_enabled!("sse"), information.sse());
        assert_eq!(cfg_feature_enabled!("sse2"), information.sse2());
        assert_eq!(cfg_feature_enabled!("sse3"), information.sse3());
        assert_eq!(cfg_feature_enabled!("ssse3"), information.ssse3());
        assert_eq!(cfg_feature_enabled!("sse4.1"), information.sse4_1());
        assert_eq!(cfg_feature_enabled!("sse4.2"), information.sse4_2());
        assert_eq!(cfg_feature_enabled!("sse4a"), information.sse4a());
        assert_eq!(cfg_feature_enabled!("avx"), information.avx());
        assert_eq!(cfg_feature_enabled!("avx2"), information.avx2());
        assert_eq!(cfg_feature_enabled!("avx512f"), information.avx512f());
        assert_eq!(cfg_feature_enabled!("avx512cd"), information.avx512cd());
        assert_eq!(cfg_feature_enabled!("avx512er"), information.avx512er());
        assert_eq!(cfg_feature_enabled!("avx512pf"), information.avx512pf());
        assert_eq!(cfg_feature_enabled!("avx512bw"), information.avx512bw());
        assert_eq!(cfg_feature_enabled!("avx512dq"), information.avx512dq());
        assert_eq!(cfg_feature_enabled!("avx512vl"), information.avx512vl());
        assert_eq!(
            cfg_feature_enabled!("avx512ifma"),
            information.avx512_ifma()
        );
        assert_eq!(
            cfg_feature_enabled!("avx512vbmi"),
            information.avx512_vbmi()
        );
        assert_eq!(
            cfg_feature_enabled!("avx512vpopcntdq"),
            information.avx512_vpopcntdq()
        );
        assert_eq!(cfg_feature_enabled!("fma"), information.fma());
        assert_eq!(cfg_feature_enabled!("bmi"), information.bmi1());
        assert_eq!(cfg_feature_enabled!("bmi2"), information.bmi2());
        assert_eq!(cfg_feature_enabled!("popcnt"), information.popcnt());
        assert_eq!(cfg_feature_enabled!("abm"), information.lzcnt());
        assert_eq!(cfg_feature_enabled!("tbm"), information.tbm());
        assert_eq!(cfg_feature_enabled!("lzcnt"), information.lzcnt());
        assert_eq!(cfg_feature_enabled!("xsave"), information.xsave());
        assert_eq!(cfg_feature_enabled!("xsaveopt"), information.xsaveopt());
        assert_eq!(
            cfg_feature_enabled!("xsavec"),
            information.xsavec_and_xrstor()
        );
        assert_eq!(
            cfg_feature_enabled!("xsaves"),
            information.xsaves_xrstors_and_ia32_xss()
        );
    }
}
