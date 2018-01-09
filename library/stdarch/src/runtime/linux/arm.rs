//! Run-time feature detection for ARM Aarch32 on Linux in `stdsimd`.

use super::cpuinfo::CpuInfo;
use coresimd::__vendor_runtime::__runtime::arch::{HasFeature, __Feature};

/// Is the CPU known to have a broken NEON unit?
///
/// See https://crbug.com/341598.
fn has_broken_neon(cpuinfo: &CpuInfo) -> bool {
    cpuinfo.field("CPU implementer") == "0x51"
        && cpuinfo.field("CPU architecture") == "7"
        && cpuinfo.field("CPU variant") == "0x1"
        && cpuinfo.field("CPU part") == "0x04d"
        && cpuinfo.field("CPU revision") == "0"
}

impl HasFeature for CpuInfo {
    fn has_feature(&mut self, x: &__Feature) -> bool {
        use self::__Feature::*;
        match *x {
            neon => {
                self.field("Features").has("neon") && !has_broken_neon(self)
            }
            pmull => self.field("Features").has("pmull"),
        }
    }
}
