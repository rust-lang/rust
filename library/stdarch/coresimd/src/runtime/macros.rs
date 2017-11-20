//! Run-time feature detection macros.

/// Is a feature supported by the host CPU?
///
/// This macro performs run-time feature detection. It returns true if the host
/// CPU in which the binary is running on supports a particular feature.
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
                __unstable_detect_feature!($name)
            }
        }
    )
}

/// In all unsupported architectures using the macro is an error
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64",
              target_arch = "arm", target_arch = "aarch64")))]
#[macro_export]
#[doc(hidden)]
macro_rules! __unstable_detect_feature {
    ($t:tt) => { compile_error!(concat!("unknown target feature: ", $t)) };
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_macros() {
        assert!(cfg_feature_enabled!("sse"));
    }
}
