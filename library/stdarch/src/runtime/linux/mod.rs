//! Run-time feature detection for ARM on linux
mod cpuinfo;
pub use self::cpuinfo::CpuInfo;

use super::__Feature;

pub trait FeatureQuery {
    fn has_feature(&mut self, x: &__Feature) -> bool;
}

fn detect_features_impl<T: FeatureQuery>(x: T) -> usize {
    #[cfg(target_arch = "arm")]
    {
        super::arm::detect_features(x)
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::detect_features(x)
    }
}

/// Detects ARM features:
pub fn detect_features() -> usize {
    // FIXME: use libc::getauxval, and if that fails /proc/auxv
    // Try to read /proc/cpuinfo
    if let Ok(v) = cpuinfo::CpuInfo::new() {
        return detect_features_impl(v);
    }
    // Otherwise all features are disabled
    0
}
