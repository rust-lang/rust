//! This module implements minimal run-time feature detection for x86.
//!
//! The features are detected using the `detect_features` function below.
//! This function uses the CPUID instruction to read the feature flags from the
//! CPU and encodes them in a `usize` where each bit position represents
//! whether a feature is available (bit is set) or unavailable (bit is cleared).
//!
//! The enum `Feature` is used to map bit positions to feature names, and the
//! the `__crate::detect::check_for!` macro is used to map string literals (e.g.,
//! "avx") to these bit positions (e.g., `Feature::avx`).
//!
//! The run-time feature detection is performed by the
//! `__crate::detect::check_for(Feature) -> bool` function. On its first call,
//! this functions queries the CPU for the available features and stores them
//! in a global `AtomicUsize` variable. The query is performed by just checking
//! whether the feature bit in this global variable is set or cleared.

/// A macro to test at *runtime* whether a CPU feature is available on
/// x86/x86-64 platforms.
///
/// This macro is provided in the standard library and will detect at runtime
/// whether the specified CPU feature is detected. This does **not** resolve at
/// compile time unless the specified feature is already enabled for the entire
/// crate. Runtime detection currently relies mostly on the `cpuid` instruction.
///
/// This macro only takes one argument which is a string literal of the feature
/// being tested for. The feature names supported are the lowercase versions of
/// the ones defined by Intel in [their documentation][docs].
///
/// ## Supported arguments
///
/// This macro supports the same names that `#[target_feature]` supports. Unlike
/// `#[target_feature]`, however, this macro does not support names separated
/// with a comma. Instead testing for multiple features must be done through
/// separate macro invocations for now.
///
/// Supported arguments are:
///
/// * `"aes"`
/// * `"pclmulqdq"`
/// * `"rdrand"`
/// * `"rdseed"`
/// * `"tsc"`
/// * `"mmx"`
/// * `"sse"`
/// * `"sse2"`
/// * `"sse3"`
/// * `"ssse3"`
/// * `"sse4.1"`
/// * `"sse4.2"`
/// * `"sse4a"`
/// * `"sha"`
/// * `"avx"`
/// * `"avx2"`
/// * `"avx512f"`
/// * `"avx512cd"`
/// * `"avx512er"`
/// * `"avx512pf"`
/// * `"avx512bw"`
/// * `"avx512dq"`
/// * `"avx512vl"`
/// * `"avx512ifma"`
/// * `"avx512vbmi"`
/// * `"avx512vpopcntdq"`
/// * `"f16c"`
/// * `"fma"`
/// * `"bmi1"`
/// * `"bmi2"`
/// * `"abm"`
/// * `"lzcnt"`
/// * `"tbm"`
/// * `"popcnt"`
/// * `"fxsr"`
/// * `"xsave"`
/// * `"xsaveopt"`
/// * `"xsaves"`
/// * `"xsavec"`
/// * `"adx"`
/// * `"rtm"`
///
/// [docs]: https://software.intel.com/sites/landingpage/IntrinsicsGuide
#[macro_export]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[allow_internal_unstable(stdsimd_internal,stdsimd)]
macro_rules! is_x86_feature_detected {
    ("aes") => {
        cfg!(target_feature = "aes") || $crate::detect::check_for(
            $crate::detect::Feature::aes)  };
    ("pclmulqdq") => {
        cfg!(target_feature = "pclmulqdq") || $crate::detect::check_for(
            $crate::detect::Feature::pclmulqdq)  };
    ("rdrand") => {
        cfg!(target_feature = "rdrand") || $crate::detect::check_for(
            $crate::detect::Feature::rdrand)  };
    ("rdseed") => {
        cfg!(target_feature = "rdseed") || $crate::detect::check_for(
            $crate::detect::Feature::rdseed)  };
    ("tsc") => {
        cfg!(target_feature = "tsc") || $crate::detect::check_for(
            $crate::detect::Feature::tsc)  };
    ("mmx") => {
        cfg!(target_feature = "mmx") || $crate::detect::check_for(
            $crate::detect::Feature::mmx)  };
    ("sse") => {
        cfg!(target_feature = "sse") || $crate::detect::check_for(
            $crate::detect::Feature::sse)  };
    ("sse2") => {
        cfg!(target_feature = "sse2") || $crate::detect::check_for(
            $crate::detect::Feature::sse2)
    };
    ("sse3") => {
        cfg!(target_feature = "sse3") || $crate::detect::check_for(
            $crate::detect::Feature::sse3)
    };
    ("ssse3") => {
        cfg!(target_feature = "ssse3") || $crate::detect::check_for(
            $crate::detect::Feature::ssse3)
    };
    ("sse4.1") => {
        cfg!(target_feature = "sse4.1") || $crate::detect::check_for(
            $crate::detect::Feature::sse4_1)
    };
    ("sse4.2") => {
        cfg!(target_feature = "sse4.2") || $crate::detect::check_for(
            $crate::detect::Feature::sse4_2)
    };
    ("sse4a") => {
        cfg!(target_feature = "sse4a") || $crate::detect::check_for(
            $crate::detect::Feature::sse4a)
    };
    ("sha") => {
        cfg!(target_feature = "sha") || $crate::detect::check_for(
            $crate::detect::Feature::sha)
    };
    ("avx") => {
        cfg!(target_feature = "avx") || $crate::detect::check_for(
            $crate::detect::Feature::avx)
    };
    ("avx2") => {
        cfg!(target_feature = "avx2") || $crate::detect::check_for(
            $crate::detect::Feature::avx2)
    };
    ("avx512f") => {
        cfg!(target_feature = "avx512f") || $crate::detect::check_for(
            $crate::detect::Feature::avx512f)
    };
    ("avx512cd") => {
        cfg!(target_feature = "avx512cd") || $crate::detect::check_for(
            $crate::detect::Feature::avx512cd)
    };
    ("avx512er") => {
        cfg!(target_feature = "avx512er") || $crate::detect::check_for(
            $crate::detect::Feature::avx512er)
    };
    ("avx512pf") => {
        cfg!(target_feature = "avx512pf") || $crate::detect::check_for(
            $crate::detect::Feature::avx512pf)
    };
    ("avx512bw") => {
        cfg!(target_feature = "avx512bw") || $crate::detect::check_for(
            $crate::detect::Feature::avx512bw)
    };
    ("avx512dq") => {
        cfg!(target_feature = "avx512dq") || $crate::detect::check_for(
            $crate::detect::Feature::avx512dq)
    };
    ("avx512vl") => {
        cfg!(target_Feature = "avx512vl") || $crate::detect::check_for(
            $crate::detect::Feature::avx512vl)
    };
    ("avx512ifma") => {
        cfg!(target_feature = "avx512ifma") || $crate::detect::check_for(
            $crate::detect::Feature::avx512_ifma)
    };
    ("avx512vbmi") => {
        cfg!(target_feature = "avx512vbmi") || $crate::detect::check_for(
            $crate::detect::Feature::avx512_vbmi)
    };
    ("avx512vpopcntdq") => {
        cfg!(target_feature = "avx512vpopcntdq") || $crate::detect::check_for(
            $crate::detect::Feature::avx512_vpopcntdq)
    };
    ("f16c") => {
        cfg!(target_feature = "f16c") || $crate::detect::check_for(
            $crate::detect::Feature::f16c)
    };
    ("fma") => {
        cfg!(target_feature = "fma") || $crate::detect::check_for(
            $crate::detect::Feature::fma)
    };
    ("bmi1") => {
        cfg!(target_feature = "bmi1") || $crate::detect::check_for(
            $crate::detect::Feature::bmi)
    };
    ("bmi2") => {
        cfg!(target_feature = "bmi2") || $crate::detect::check_for(
            $crate::detect::Feature::bmi2)
    };
    ("abm") => {
        cfg!(target_feature = "abm") || $crate::detect::check_for(
            $crate::detect::Feature::abm)
    };
    ("lzcnt") => {
        cfg!(target_feature = "lzcnt") || $crate::detect::check_for(
            $crate::detect::Feature::abm)
    };
    ("tbm") => {
        cfg!(target_feature = "tbm") || $crate::detect::check_for(
            $crate::detect::Feature::tbm)
    };
    ("popcnt") => {
        cfg!(target_feature = "popcnt") || $crate::detect::check_for(
            $crate::detect::Feature::popcnt)
    };
    ("fxsr") => {
        cfg!(target_feature = "fxsr") || $crate::detect::check_for(
            $crate::detect::Feature::fxsr)
    };
    ("xsave") => {
        cfg!(target_feature = "xsave") || $crate::detect::check_for(
            $crate::detect::Feature::xsave)
    };
    ("xsaveopt") => {
        cfg!(target_feature = "xsaveopt") || $crate::detect::check_for(
            $crate::detect::Feature::xsaveopt)
    };
    ("xsaves") => {
        cfg!(target_feature = "xsaves") || $crate::detect::check_for(
            $crate::detect::Feature::xsaves)
    };
    ("xsavec") => {
        cfg!(target_feature = "xsavec") || $crate::detect::check_for(
            $crate::detect::Feature::xsavec)
    };
    ("cmpxchg16b") => {
        cfg!(target_feature = "cmpxchg16b") || $crate::detect::check_for(
            $crate::detect::Feature::cmpxchg16b)
    };
    ("adx") => {
        cfg!(target_feature = "adx") || $crate::detect::check_for(
            $crate::detect::Feature::adx)
    };
    ("rtm") => {
        cfg!(target_feature = "rtm") || $crate::detect::check_for(
            $crate::detect::Feature::rtm)
    };
    ($t:tt,) => {
        is_x86_feature_detected!($t);
    };
    ($t:tt) => {
        compile_error!(concat!("unknown target feature: ", $t))
    };
}

/// X86 CPU Feature enum. Each variant denotes a position in a bitset for a
/// particular feature.
///
/// This is an unstable implementation detail subject to change.
#[allow(non_camel_case_types)]
#[repr(u8)]
#[doc(hidden)]
#[unstable(feature = "stdsimd_internal", issue = "0")]
pub enum Feature {
    /// AES (Advanced Encryption Standard New Instructions AES-NI)
    aes,
    /// CLMUL (Carry-less Multiplication)
    pclmulqdq,
    /// RDRAND
    rdrand,
    /// RDSEED
    rdseed,
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
    /// SHA
    sha,
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
    /// F16C (Conversions between IEEE-754 `binary16` and `binary32` formats)
    f16c,
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
    /// FXSR (Floating-point context fast save and restore)
    fxsr,
    /// XSAVE (Save Processor Extended States)
    xsave,
    /// XSAVEOPT (Save Processor Extended States Optimized)
    xsaveopt,
    /// XSAVES (Save Processor Extended States Supervisor)
    xsaves,
    /// XSAVEC (Save Processor Extended States Compacted)
    xsavec,
    /// CMPXCH16B, a 16-byte compare-and-swap instruction
    cmpxchg16b,
    /// ADX, Intel ADX (Multi-Precision Add-Carry Instruction Extensions)
    adx,
    /// RTM, Intel (Restricted Transactional Memory)
    rtm,
}
