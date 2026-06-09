//! Run-time feature detection for ARM on Linux.

use super::{auxvec, cpuinfo};
use crate::detect::{bit, cache, Feature};

/// Performs run-time feature detection.
#[inline]
pub fn check_for(x: Feature) -> bool {
    cache::test(x as u32, detect_features)
}

/// Try to read the features from the auxiliary vector, and if that fails, try
/// to read them from /proc/cpuinfo.
fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let enable_feature = |value: &mut cache::Initializer, f, enable| {
        if enable {
            value.set(f as u32);
        }
    };

    // The values are part of the platform-specific [asm/hwcap.h][hwcap]
    //
    // [hwcap]: https://github.com/torvalds/linux/blob/master/arch/arm64/include/uapi/asm/hwcap.h
    if let Ok(auxv) = auxvec::auxv() {
        enable_feature(&mut value, Feature::neon, bit::test(auxv.hwcap, 12));
        enable_feature(&mut value, Feature::pmull, bit::test(auxv.hwcap2, 1));
        return value;
    }

    if let Ok(c) = cpuinfo::CpuInfo::new() {
        enable_feature(
            &mut value,
            Feature::neon,
            c.field("Features").has("neon") && !has_broken_neon(&c),
        );
        enable_feature(&mut value, Feature::pmull, c.field("Features").has("pmull"));
        return value;
    }
    value
}

/// Is the CPU known to have a broken NEON unit?
///
/// See https://crbug.com/341598.
fn has_broken_neon(cpuinfo: &cpuinfo::CpuInfo) -> bool {
    cpuinfo.field("CPU implementer") == "0x51"
        && cpuinfo.field("CPU architecture") == "7"
        && cpuinfo.field("CPU variant") == "0x1"
        && cpuinfo.field("CPU part") == "0x04d"
        && cpuinfo.field("CPU revision") == "0"
}
