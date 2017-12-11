//! Run-time feature detection on ARM Aarch32.

use super::{bit, linux};

#[macro_export]
#[doc(hidden)]
macro_rules! __unstable_detect_feature {
    ("neon") => {
        $crate::vendor::__unstable_detect_feature($crate::vendor::__Feature::neon{})
    };
    ("pmull") => {
        $crate::vendor::__unstable_detect_feature($crate::vendor::__Feature::pmull{})
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
pub enum __Feature {
    /// ARM Advanced SIMD (NEON) - Aarch32
    neon,
    /// Polynomial Multiply
    pmull,
}

pub fn detect_features<T: linux::FeatureQuery>(mut x: T) -> usize {
    let mut value: usize = 0;
    {
        let mut enable_feature = |f| {
            if x.has_feature(&f) {
                value = bit::set(value, f as u32);
            }
        };
        enable_feature(__Feature::neon);
        enable_feature(__Feature::pmull);
    }
    value
}

/// Probe the ELF Auxiliary vector for hardware capabilities
///
/// The values are part of the platform-specific [asm/hwcap.h][hwcap]
///
/// [hwcap]: https://github.com/torvalds/linux/blob/master/arch/arm64/include/uapi/asm/hwcap.h
impl linux::FeatureQuery for linux::AuxVec {
    fn has_feature(&mut self, x: &__Feature) -> bool {
        use self::__Feature::*;
        match *x {
            neon => self.lookup(linux::AT::HWCAP)
                .map(|caps| caps & (1 << 12) != 0)
                .unwrap_or(false),
            pmull => self.lookup(linux::AT::HWCAP2)
                .map(|caps| caps & (1 << 1) != 0)
                .unwrap_or(false),
        }
    }
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

impl linux::FeatureQuery for linux::CpuInfo {
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
