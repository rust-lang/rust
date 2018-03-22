//! This module implements minimal run-time feature detection for x86.
//!
//! The features are detected using the `detect_features` function below.
//! This function uses the CPUID instruction to read the feature flags from the
//! CPU and encodes them in an `usize` where each bit position represents
//! whether a feature is available (bit is set) or unavaiable (bit is cleared).
//!
//! The enum `Feature` is used to map bit positions to feature names, and the
//! the `__crate::arch::detect::check_for!` macro is used to map string literals (e.g.
//! "avx") to these bit positions (e.g. `Feature::avx`).
//!
//!
//! The run-time feature detection is performed by the
//! `__crate::arch::detect::check_for(Feature) -> bool` function. On its first call,
//! this functions queries the CPU for the available features and stores them
//! in a global `AtomicUsize` variable. The query is performed by just checking
//! whether the feature bit in this global variable is set or cleared.

#[macro_export]
#[unstable(feature = "stdsimd", issue = "0")]
macro_rules! is_x86_feature_detected {
    ("aes") => {
        cfg!(target_feature = "aes") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::aes)  };
    ("pclmulqdq") => {
        cfg!(target_feature = "pclmulqdq") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::pclmulqdq)  };
    ("rdrand") => {
        cfg!(target_feature = "rdrand") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::rdrand)  };
    ("rdseed") => {
        cfg!(target_feature = "rdseed") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::rdseed)  };
    ("tsc") => {
        cfg!(target_feature = "tsc") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::tsc)  };
    ("mmx") => {
        cfg!(target_feature = "mmx") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::mmx)  };
    ("sse") => {
        cfg!(target_feature = "sse") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::sse)  };
    ("sse2") => {
        cfg!(target_feature = "sse2") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::sse2)
    };
    ("sse3") => {
        cfg!(target_feature = "sse3") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::sse3)
    };
    ("ssse3") => {
        cfg!(target_feature = "ssse3") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::ssse3)
    };
    ("sse4.1") => {
        cfg!(target_feature = "sse4.1") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::sse4_1)
    };
    ("sse4.2") => {
        cfg!(target_feature = "sse4.2") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::sse4_2)
    };
    ("sse4a") => {
        cfg!(target_feature = "sse4a") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::sse4a)
    };
    ("sha") => {
        cfg!(target_feature = "sha") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::sha)
    };
    ("avx") => {
        cfg!(target_feature = "avx") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::avx)
    };
    ("avx2") => {
        cfg!(target_feature = "avx2") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::avx2)
    };
    ("avx512f") => {
        cfg!(target_feature = "avx512f") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::avx512f)
    };
    ("avx512cd") => {
        cfg!(target_feature = "avx512cd") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::avx512cd)
    };
    ("avx512er") => {
        cfg!(target_feature = "avx512er") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::avx512er)
    };
    ("avx512pf") => {
        cfg!(target_feature = "avx512pf") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::avx512pf)
    };
    ("avx512bw") => {
        cfg!(target_feature = "avx512bw") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::avx512bw)
    };
    ("avx512dq") => {
        cfg!(target_feature = "avx512dq") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::avx512dq)
    };
    ("avx512vl") => {
        cfg!(target_Feature = "avx512vl") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::avx512vl)
    };
    ("avx512ifma") => {
        cfg!(target_feature = "avx512ifma") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::avx512_ifma)
    };
    ("avx512vbmi") => {
        cfg!(target_feature = "avx512vbmi") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::avx512_vbmi)
    };
    ("avx512vpopcntdq") => {
        cfg!(target_feature = "avx512vpopcntdq") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::avx512_vpopcntdq)
    };
    ("fma") => {
        cfg!(target_feature = "fma") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::fma)
    };
    ("bmi1") => {
        cfg!(target_feature = "bmi1") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::bmi)
    };
    ("bmi2") => {
        cfg!(target_feature = "bmi2") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::bmi2)
    };
    ("abm") => {
        cfg!(target_feature = "abm") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::abm)
    };
    ("lzcnt") => {
        cfg!(target_feature = "lzcnt") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::abm)
    };
    ("tbm") => {
        cfg!(target_feature = "tbm") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::tbm)
    };
    ("popcnt") => {
        cfg!(target_feature = "popcnt") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::popcnt)
    };
    ("fxsr") => {
        cfg!(target_feature = "fxsr") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::fxsr)
    };
    ("xsave") => {
        cfg!(target_feature = "xsave") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::xsave)
    };
    ("xsaveopt") => {
        cfg!(target_feature = "xsaveopt") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::xsaveopt)
    };
    ("xsaves") => {
        cfg!(target_feature = "xsaves") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::xsaves)
    };
    ("xsavec") => {
        cfg!(target_feature = "xsavec") || $crate::arch::detect::check_for(
            $crate::arch::detect::Feature::xsavec)
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
}
