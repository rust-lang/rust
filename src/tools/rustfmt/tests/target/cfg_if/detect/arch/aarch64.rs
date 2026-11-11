//! Aarch64 run-time features.

/// Checks if `aarch64` feature is enabled.
#[macro_export]
#[unstable(feature = "stdsimd", issue = "27731")]
#[allow_internal_unstable(stdsimd_internal, stdsimd)]
macro_rules! is_aarch64_feature_detected {
    ("neon") => {
        // FIXME: this should be removed once we rename Aarch64 neon to asimd
        cfg!(target_feature = "neon") || $crate::detect::check_for($crate::detect::Feature::asimd)
    };
    ("asimd") => {
        cfg!(target_feature = "neon") || $crate::detect::check_for($crate::detect::Feature::asimd)
    };
    ("pmull") => {
        cfg!(target_feature = "pmull") || $crate::detect::check_for($crate::detect::Feature::pmull)
    };
    ("fp") => {
        cfg!(target_feature = "fp") || $crate::detect::check_for($crate::detect::Feature::fp)
    };
    ("fp16") => {
        cfg!(target_feature = "fp16") || $crate::detect::check_for($crate::detect::Feature::fp16)
    };
    ("sve") => {
        cfg!(target_feature = "sve") || $crate::detect::check_for($crate::detect::Feature::sve)
    };
    ("crc") => {
        cfg!(target_feature = "crc") || $crate::detect::check_for($crate::detect::Feature::crc)
    };
    ("crypto") => {
        cfg!(target_feature = "crypto")
            || $crate::detect::check_for($crate::detect::Feature::crypto)
    };
    ("lse") => {
        cfg!(target_feature = "lse") || $crate::detect::check_for($crate::detect::Feature::lse)
    };
    ("rdm") => {
        cfg!(target_feature = "rdm") || $crate::detect::check_for($crate::detect::Feature::rdm)
    };
    ("rcpc") => {
        cfg!(target_feature = "rcpc") || $crate::detect::check_for($crate::detect::Feature::rcpc)
    };
    ("dotprod") => {
        cfg!(target_feature = "dotprod")
            || $crate::detect::check_for($crate::detect::Feature::dotprod)
    };
    ("ras") => {
        compile_error!("\"ras\" feature cannot be detected at run-time")
    };
    ("v8.1a") => {
        compile_error!("\"v8.1a\" feature cannot be detected at run-time")
    };
    ("v8.2a") => {
        compile_error!("\"v8.2a\" feature cannot be detected at run-time")
    };
    ("v8.3a") => {
        compile_error!("\"v8.3a\" feature cannot be detected at run-time")
    };
    ($t:tt,) => {
        is_aarch64_feature_detected!($t);
    };
    ($t:tt) => {
        compile_error!(concat!("unknown aarch64 target feature: ", $t))
    };
}

/// ARM Aarch64 CPU Feature enum. Each variant denotes a position in a bitset
/// for a particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
#[unstable(feature = "stdsimd_internal", issue = "0")]
pub enum Feature {
    /// ARM Advanced SIMD (ASIMD)
    asimd,
    /// Polynomial Multiply
    pmull,
    /// Floating point support
    fp,
    /// Half-float support.
    fp16,
    /// Scalable Vector Extension (SVE)
    sve,
    /// CRC32 (Cyclic Redundancy Check)
    crc,
    /// Crypto: AES + PMULL + SHA1 + SHA2
    crypto,
    /// Atomics (Large System Extension)
    lse,
    /// Rounding Double Multiply (ASIMDRDM)
    rdm,
    /// Release consistent Processor consistent (RcPc)
    rcpc,
    /// Vector Dot-Product (ASIMDDP)
    dotprod,
}
