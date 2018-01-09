//! Run-time feature detection macros.

/// Is a feature supported by the host CPU?
///
/// This macro performs run-time feature detection in `stdsimd`. It returns
/// true if the host CPU in which the binary is running on supports a
/// particular feature.
#[macro_export]
macro_rules! cfg_feature_enabled {
    ($name:tt) => (
        {
            #[cfg(target_feature = $name)]
            {
                true
            }
            #[cfg(not(target_feature = $name))]
            {
                __unstable_detect_feature!($name,
                                           $crate::__vendor_runtime::__unstable_detect_feature)
            }
        }
    )
}
