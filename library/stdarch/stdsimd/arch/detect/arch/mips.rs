//! Run-time feature detection on MIPS.

#[macro_export]
#[unstable(feature = "stdsimd", issue = "0")]
macro_rules! is_mips_feature_detected {
    ("msa") => {
        cfg!(target_feature = "msa") ||
            $crate::arch::detect::check_for($crate::arch::detect::Feature::msa)
    };
    ($t:tt) => { compile_error!(concat!("unknown mips target feature: ", $t)) };
}

/// MIPS CPU Feature enum. Each variant denotes a position in a bitset for a
/// particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum Feature {
    /// MIPS SIMD Architecture (MSA)
    msa,
}
