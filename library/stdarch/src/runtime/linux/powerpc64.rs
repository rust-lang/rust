//! Run-time feature detection for PowerPC64 on Linux in `stdsimd`.

use super::cpuinfo::CpuInfo;
use coresimd::__vendor_runtime::__runtime::arch::{HasFeature, __Feature};

/// Check for altivec support only
///
/// PowerPC's /proc/cpuinfo lacks a proper Feature field,
/// but `altivec` support is indicated in the `cpu` field.
impl HasFeature for CpuInfo {
    fn has_feature(&mut self, x: &__Feature) -> bool {
        use self::__Feature::*;
        match *x {
            altivec => self.field("cpu").has("altivec"),
            _ => false,
        }
    }
}
