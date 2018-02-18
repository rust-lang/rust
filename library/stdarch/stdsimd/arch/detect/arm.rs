//! Run-time feature detection on ARM Aarch32.

use super::bit;
use super::cache;
use super::linux;

#[macro_export]
macro_rules! is_target_feature_detected {
    ("neon") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::neon)
    };
    ("pmull") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::pmull)
    };
    ($t:tt) => { compile_error!(concat!("unknown arm target feature: ", $t)) };
}

/// ARM CPU Feature enum. Each variant denotes a position in a bitset for a
/// particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum Feature {
    /// ARM Advanced SIMD (NEON) - Aarch32
    neon,
    /// Polynomial Multiply
    pmull,
}

pub fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    fill_features(&mut value);
    return value
}

fn fill_features(value: &mut cache::Initializer) {
    let mut enable_feature = |f, enable| {
        if enable {
            value.set(f as u32);
        }
    };

    // The values are part of the platform-specific [asm/hwcap.h][hwcap]
    //
    // [hwcap]: https://github.com/torvalds/linux/blob/master/arch/arm64/include/uapi/asm/hwcap.h
    if let Ok(auxv) = linux::auxv() {
        enable_feature(Feature::neon, bit::test(auxv.hwcap, 12));
        enable_feature(Feature::pmull, bit::test(auxv.hwcap2, 1));
        return
    }

    if let Ok(c) = linux::CpuInfo::new() {
        enable_feature(Feature::neon, c.field("Features").has("neon") &&
            !has_broken_neon(&c));
        enable_feature(Feature::pmull, c.field("Features").has("pmull"));
        return
    }

    /// Is the CPU known to have a broken NEON unit?
    ///
    /// See https://crbug.com/341598.
    fn has_broken_neon(cpuinfo: &linux::CpuInfo) -> bool {
        cpuinfo.field("CPU implementer") == "0x51"
            && cpuinfo.field("CPU architecture") == "7"
            && cpuinfo.field("CPU variant") == "0x1"
            && cpuinfo.field("CPU part") == "0x04d"
            && cpuinfo.field("CPU revision") == "0"
    }
}
