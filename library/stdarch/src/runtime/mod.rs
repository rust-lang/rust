//! Run-time feature detection

#[cfg(all(target_os = "linux",
          any(target_arch = "arm", target_arch = "aarch64",
              target_arch = "powerpc64")))]
mod linux;

#[macro_use]
mod macros;

/// Run-time feature detection exposed by `stdsimd`.
pub mod std {
    // The x86/x86_64 run-time from `coresimd` is re-exported as is.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use coresimd::__vendor_runtime::*;

    #[cfg(all(target_os = "linux",
              any(target_arch = "arm", target_arch = "aarch64",
                  target_arch = "powerpc64")))]
    pub use super::linux::{detect_features, __Feature};

    /// Performs run-time feature detection.
    ///
    /// For those platforms in which run-time detection differs between `core`
    /// and `std`.
    #[cfg(all(target_os = "linux",
              any(target_arch = "arm", target_arch = "aarch64",
                  target_arch = "powerpc64")))]
    #[doc(hidden)]
    pub fn __unstable_detect_feature(x: __Feature) -> bool {
        ::coresimd::__vendor_runtime::__runtime::cache::test(
            x as u32,
            detect_features,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn detect_feature() {
        println!("avx {}", cfg_feature_enabled!("avx"));
    }

    #[cfg(all(target_arch = "arm", target_os = "linux"))]
    #[test]
    fn detect_feature() {
        println!("neon {}", cfg_feature_enabled!("neon"));
        println!("pmull {}", cfg_feature_enabled!("pmull"));
    }

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    #[test]
    fn detect_feature() {
        println!("asimd {}", cfg_feature_enabled!("asimd"));
        println!("pmull {}", cfg_feature_enabled!("pmull"));
    }

    #[cfg(all(target_arch = "powerpc64", target_os = "linux"))]
    #[test]
    fn detect_feature() {
        println!("altivec {}", cfg_feature_enabled!("altivec"));
        println!("vsx {}", cfg_feature_enabled!("vsx"));
        println!("power8 {}", cfg_feature_enabled!("power8"));
    }
}
