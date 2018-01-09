//! Run-time feature detection for ARM Aarch64 on Linux in `stdsimd`.

use super::cpuinfo::CpuInfo;
use coresimd::__vendor_runtime::__runtime::arch::{HasFeature, __Feature};

impl HasFeature for CpuInfo {
    fn has_feature(&mut self, x: &__Feature) -> bool {
        use self::__Feature::*;
        match *x {
            asimd => self.field("Features").has("asimd"),
            pmull => self.field("Features").has("pmull"),
        }
    }
}
