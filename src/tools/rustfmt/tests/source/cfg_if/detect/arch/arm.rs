//! Run-time feature detection on ARM Aarch32.

/// Checks if `arm` feature is enabled.
#[macro_export]
#[unstable(feature = "stdsimd", issue = "27731")]
#[allow_internal_unstable(stdsimd_internal,stdsimd)]
macro_rules! is_arm_feature_detected {
    ("neon") => {
        cfg!(target_feature = "neon") ||
            $crate::detect::check_for($crate::detect::Feature::neon)
    };
    ("pmull") => {
        cfg!(target_feature = "pmull") ||
            $crate::detect::check_for($crate::detect::Feature::pmull)
    };
    ("v7") => { compile_error!("\"v7\" feature cannot be detected at run-time") };
    ("vfp2") => { compile_error!("\"vfp2\" feature cannot be detected at run-time") };
    ("vfp3") => { compile_error!("\"vfp3\" feature cannot be detected at run-time") };
    ("vfp4") => { compile_error!("\"vfp4\" feature cannot be detected at run-time") };
    ($t:tt,) => {
        is_arm_feature_detected!($t);
    };
    ($t:tt) => { compile_error!(concat!("unknown arm target feature: ", $t)) };
}

/// ARM CPU Feature enum. Each variant denotes a position in a bitset for a
/// particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
#[unstable(feature = "stdsimd_internal", issue = "0")]
pub enum Feature {
    /// ARM Advanced SIMD (NEON) - Aarch32
    neon,
    /// Polynomial Multiply
    pmull,
}
