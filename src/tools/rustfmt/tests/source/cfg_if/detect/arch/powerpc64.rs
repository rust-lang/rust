//! Run-time feature detection on PowerPC64.

/// Checks if `powerpc64` feature is enabled.
#[macro_export]
#[unstable(feature = "stdsimd", issue = "27731")]
#[allow_internal_unstable(stdsimd_internal,stdsimd)]
macro_rules! is_powerpc64_feature_detected {
    ("altivec") => {
        cfg!(target_feature = "altivec") ||
            $crate::detect::check_for($crate::detect::Feature::altivec)
    };
    ("vsx") => {
        cfg!(target_feature = "vsx") ||
            $crate::detect::check_for($crate::detect::Feature::vsx)
    };
    ("power8") => {
        cfg!(target_feature = "power8") ||
            $crate::detect::check_for($crate::detect::Feature::power8)
    };
    ($t:tt,) => {
        is_powerpc64_feature_detected!($t);
    };
    ($t:tt) => { compile_error!(concat!("unknown powerpc64 target feature: ", $t)) };
}


/// PowerPC64 CPU Feature enum. Each variant denotes a position in a bitset
/// for a particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
#[unstable(feature = "stdsimd_internal", issue = "0")]
pub enum Feature {
    /// Altivec
    altivec,
    /// VSX
    vsx,
    /// Power8
    power8,
}
