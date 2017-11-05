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
use std::sync::atomic::{AtomicUsize, Ordering};

/// This macro maps the string-literal feature names to values of the
/// `__Feature` enum at compile-time. The feature names used are the same as
/// those of rustc `target_feature` and `cfg_target_feature` features.
///
/// PLESE: do not use this, it is an implementation detail subjected to change.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_export]
#[doc(hidden)]
macro_rules! __unstable_detect_feature {
    ("sse") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::sse{})  };
    ("sse2") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::sse2{})
    };
    ("sse3") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::sse3{})
    };
    ("ssse3") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::ssse3{})
    };
    ("sse4.1") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::sse4_1{})
    };
    ("sse4.2") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::sse4_2{})
    };
    ("sse4a") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::sse4a{})
    };
    ("avx") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::avx{})
    };
    ("avx2") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::avx2{})
    };
    ("avx512f") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::avx512f{})
    };
    ("avx512cd") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::avx512cd{})
    };
    ("avx512er") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::avx512er{})
    };
    ("avx512pf") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::avx512pf{})
    };
    ("avx512bw") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::avx512bw{})
    };
    ("avx512dq") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::avx512dq{})
    };
    ("avx512vl") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::avx512vl{})
    };
    ("avx512ifma") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::avx512_ifma{})
    };
    ("avx512vbmi") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::avx512_vbmi{})
    };
    ("avx512vpopcntdq") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::avx512_vpopcntdq{})
    };
    ("fma") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::fma{})
    };
    ("bmi") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::bmi{})
    };
    ("bmi2") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::bmi2{})
    };
    ("abm") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::abm{})
    };
    ("lzcnt") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::abm{})
    };
    ("tbm") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::tbm{})
    };
    ("popcnt") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::popcnt{})
    };
    ("xsave") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::xsave{})
    };
    ("xsaveopt") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::xsaveopt{})
    };
    ("xsave") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::xsave{})
    };
    ("xsaveopt") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::xsaveopt{})
    };
    ("xsaves") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::xsaves{})
    };
    ("xsavec") => {
        $crate::vendor::__unstable_detect_feature(
            $crate::vendor::__Feature::xsavec{})
    };
    ($t:tt) => {
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
    /// AVX-512 VPOPCNTDQ (Vector Population Count Doubleword and Quadword)
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
    /// XSAVE (Save Processor Extended States)
    xsave,
    /// XSAVEOPT (Save Processor Extended States Optimized)
    xsaveopt,
    /// XSAVES (Save Processor Extended States Supervisor)
    xsaves,
    /// XSAVEC (Save Processor Extended States Compacted)
    xsavec,
    #[doc(hidden)] __NonExhaustive,
}

/// Sets the `bit`-th bit of `x`.
fn set_bit(x: usize, bit: u32) -> usize {
    debug_assert!(32 > bit);
    x | 1 << bit
}

/// Tests the `bit`-th bit of `x`.
fn test_bit(x: usize, bit: u32) -> bool {
    debug_assert!(32 > bit);
    x & (1 << bit) != 0
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
fn detect_features() -> usize {
    use super::cpuid::{__cpuid, __cpuid_count, has_cpuid, CpuidResult};
    use super::xsave::_xgetbv;
    let mut value: usize = 0;

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
    let (max_leaf, vendor_id) = unsafe {
        let CpuidResult {
            eax: max_leaf,
            ebx,
            ecx,
            edx,
        } = __cpuid(0);
        let vendor_id: [[u8; 4]; 3] = [
            ::std::mem::transmute(ebx),
            ::std::mem::transmute(edx),
            ::std::mem::transmute(ecx),
        ];
        let vendor_id: [u8; 12] = ::std::mem::transmute(vendor_id);
        (max_leaf, vendor_id)
    };

    if max_leaf < 1 {
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
    let (extended_features_ebx, extended_features_ecx) = if max_leaf >= 7 {
        let CpuidResult { ebx, ecx, .. } = unsafe { __cpuid(0x0000_0007_u32) };
        (ebx, ecx)
    } else {
        (0, 0) // CPUID does not support "Extended Features"
    };

    // EAX = 0x8000_0000, ECX = 0: Get Highest Extended Function Supported
    // - EAX returns the max leaf value for extended information, that is,
    // `cpuid` calls in range [0x8000_0000; u32::MAX]:
    let CpuidResult {
        eax: extended_max_leaf,
        ..
    } = unsafe { __cpuid(0x8000_0000_u32) };

    // EAX = 0x8000_0001, ECX=0: Queries "Extended Processor Info and Feature
    // Bits"
    let extended_proc_info_ecx = if extended_max_leaf >= 1 {
        let CpuidResult { ecx, .. } = unsafe { __cpuid(0x8000_0001_u32) };
        ecx
    } else {
        0
    };

    {
        // borrows value till the end of this scope:
        let mut enable = |r, rb, f| if test_bit(r as usize, rb) {
            value = set_bit(value, f as u32);
        };

        enable(proc_info_ecx, 0, __Feature::sse3);
        enable(proc_info_ecx, 9, __Feature::ssse3);
        enable(proc_info_ecx, 12, __Feature::fma);
        enable(proc_info_ecx, 19, __Feature::sse4_1);
        enable(proc_info_ecx, 20, __Feature::sse4_2);
        enable(proc_info_ecx, 23, __Feature::popcnt);
        enable(proc_info_edx, 25, __Feature::sse);
        enable(proc_info_edx, 26, __Feature::sse2);

        enable(extended_features_ebx, 3, __Feature::bmi);
        enable(extended_features_ebx, 8, __Feature::bmi2);

        // `XSAVE` and `AVX` support:
        if test_bit(proc_info_ecx as usize, 26) {
            // 0. Here the CPU supports `XSAVE`.

            // 1. Detect `OSXSAVE`, that is, whether the OS is AVX enabled and
            // supports saving the state of the AVX/AVX2 vector registers on
            // context-switches, see:
            //
            // - https://software.intel.
            // com/en-us/blogs/2011/04/14/is-avx-enabled
            // - https://hg.mozilla.
            // org/mozilla-central/file/64bab5cbb9b6/mozglue/build/SSE.cpp#l190
            let cpu_osxsave = test_bit(proc_info_ecx as usize, 27);

            // 2. The OS must have signaled the CPU that it supports saving and
            // restoring the SSE and AVX registers by setting `XCR0.SSE[1]` and
            // `XCR0.AVX[2]` to `1`.
            //
            // This is safe because the CPU supports `xsave`
            let xcr0 = unsafe { _xgetbv(0) };
            let os_avx_support = xcr0 & 6 == 6;
            let os_avx512_support = xcr0 & 224 == 224;

            if cpu_osxsave && os_avx_support {
                // Only if the OS and the CPU support saving/restoring the AVX
                // registers we enable `xsave` support:
                enable(proc_info_ecx, 26, __Feature::xsave);

                // And AVX/AVX2:
                enable(proc_info_ecx, 28, __Feature::avx);
                enable(extended_features_ebx, 5, __Feature::avx2);

                // For AVX-512 the OS also needs to support saving/restoring
                // the
                // extended state, only then we enable AVX-512 support:
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

            // Processor Extended State Enumeration Sub-leaf (EAX = 0DH, ECX =
            // 1)
            if max_leaf >= 0xd {
                let CpuidResult {
                    eax: proc_extended_state1_eax,
                    ..
                } = unsafe { __cpuid_count(0xd_u32, 1) };
                enable(proc_extended_state1_eax, 0, __Feature::xsaveopt);
                enable(proc_extended_state1_eax, 1, __Feature::xsavec);
                enable(proc_extended_state1_eax, 3, __Feature::xsaves);
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

/// This global variable is a bitset used to cache the features supported by
/// the CPU.
static FEATURES: AtomicUsize = AtomicUsize::new(::std::usize::MAX);

/// Performs run-time feature detection.
///
/// On its first invocation, it detects the CPU features and caches them
/// in the `FEATURES` global variable as an `AtomicUsize`.
///
/// It uses the `__Feature` variant to index into this variable as a bitset. If
/// the bit is set, the feature is enabled, and otherwise it is disabled.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
pub fn __unstable_detect_feature(x: __Feature) -> bool {
    if FEATURES.load(Ordering::Relaxed) == ::std::usize::MAX {
        FEATURES.store(detect_features(), Ordering::Relaxed);
    }
    test_bit(FEATURES.load(Ordering::Relaxed), x as u32)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    #[test]
    fn runtime_detection_x86_nocapture() {
        println!("sse: {:?}", cfg_feature_enabled!("sse"));
        println!("sse2: {:?}", cfg_feature_enabled!("sse2"));
        println!("sse3: {:?}", cfg_feature_enabled!("sse3"));
        println!("ssse3: {:?}", cfg_feature_enabled!("ssse3"));
        println!("sse4.1: {:?}", cfg_feature_enabled!("sse4.1"));
        println!("sse4.2: {:?}", cfg_feature_enabled!("sse4.2"));
        println!("avx: {:?}", cfg_feature_enabled!("avx"));
        println!("avx2: {:?}", cfg_feature_enabled!("avx2"));
        println!("avx512f {:?}", cfg_feature_enabled!("avx512f"));
        println!("avx512cd {:?}", cfg_feature_enabled!("avx512cd"));
        println!("avx512er {:?}", cfg_feature_enabled!("avx512er"));
        println!("avx512pf {:?}", cfg_feature_enabled!("avx512pf"));
        println!("avx512bw {:?}", cfg_feature_enabled!("avx512bw"));
        println!("avx512dq {:?}", cfg_feature_enabled!("avx512dq"));
        println!("avx512vl {:?}", cfg_feature_enabled!("avx512vl"));
        println!("avx512ifma {:?}", cfg_feature_enabled!("avx512ifma"));
        println!("avx512vbmi {:?}", cfg_feature_enabled!("avx512vbmi"));
        println!(
            "avx512vpopcntdq {:?}",
            cfg_feature_enabled!("avx512vpopcntdq")
        );
        println!("fma: {:?}", cfg_feature_enabled!("fma"));
        println!("abm: {:?}", cfg_feature_enabled!("abm"));
        println!("bmi: {:?}", cfg_feature_enabled!("bmi"));
        println!("bmi2: {:?}", cfg_feature_enabled!("bmi2"));
        println!("tbm: {:?}", cfg_feature_enabled!("tbm"));
        println!("popcnt: {:?}", cfg_feature_enabled!("popcnt"));
        println!("lzcnt: {:?}", cfg_feature_enabled!("lzcnt"));
        println!("xsave {:?}", cfg_feature_enabled!("xsave"));
        println!("xsaveopt {:?}", cfg_feature_enabled!("xsaveopt"));
        println!("xsaves {:?}", cfg_feature_enabled!("xsaves"));
        println!("xsavec {:?}", cfg_feature_enabled!("xsavec"));
    }
}
