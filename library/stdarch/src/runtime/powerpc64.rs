//! Run-time feature detection on PowerPC64.
use super::{bit, linux};

#[macro_export]
#[doc(hidden)]
macro_rules! __unstable_detect_feature {
    ("altivec") => {
        $crate::vendor::__unstable_detect_feature($crate::vendor::__Feature::altivec{})
    };
    ("vsx") => {
        $crate::vendor::__unstable_detect_feature($crate::vendor::__Feature::vsx{})
    };
    ("power8") => {
        $crate::vendor::__unstable_detect_feature($crate::vendor::__Feature::power8{})
    };
    ($t:tt) => { compile_error!(concat!("unknown PowerPC target feature: ", $t)) };
}

/// PowerPC CPU Feature enum. Each variant denotes a position in a bitset
/// for a particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum __Feature {
    /// Altivec
    altivec,
    /// VSX
    vsx,
    /// Power8
    power8,
}

pub fn detect_features<T: linux::FeatureQuery>(mut x: T) -> usize {
    let mut value: usize = 0;
    {
        let mut enable_feature = |f| {
            if x.has_feature(&f) {
                value = bit::set(value, f as u32);
            }
        };
        enable_feature(__Feature::altivec);
        enable_feature(__Feature::vsx);
        enable_feature(__Feature::power8);
    }
    value
}

/// Probe the ELF Auxiliary vector for hardware capabilities
///
/// The values are part of the platform-specific [asm/cputable.h][cputable]
///
/// [cputable]: https://github.com/torvalds/linux/blob/master/arch/powerpc/include/uapi/asm/cputable.h
impl linux::FeatureQuery for linux::AuxVec {
    fn has_feature(&mut self, x: &__Feature) -> bool {
        use self::__Feature::*;
        match *x {
            altivec => self.lookup(linux::AT::HWCAP)
                .map(|caps| caps & 0x10000000 != 0)
                .unwrap_or(false),
            vsx => self.lookup(linux::AT::HWCAP)
                .map(|caps| caps & 0x00000080 != 0)
                .unwrap_or(false),
            power8 => self.lookup(linux::AT::HWCAP2)
                .map(|caps| caps & 0x80000000 != 0)
                .unwrap_or(false),
        }
    }
}

/// Check for altivec support only
///
/// PowerPC's /proc/cpuinfo lacks a proper Feature field,
/// but `altivec` support is indicated in the `cpu` field.
impl linux::FeatureQuery for linux::CpuInfo {
    fn has_feature(&mut self, x: &__Feature) -> bool {
        use self::__Feature::*;
        match *x {
            altivec => self.field("cpu").has("altivec"),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn detect_feature() {
        println!("altivec {}", __unstable_detect_feature!("altivec"));
        println!("vsx {}", __unstable_detect_feature!("vsx"));
        println!("power8 {}", __unstable_detect_feature!("power8"));
    }
}
