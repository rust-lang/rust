//! Run-time feature detection on PowerPC64.

use super::cache;
use super::linux;

#[macro_export]
#[unstable(feature = "stdsimd", issue = "0")]
macro_rules! is_powerpc64_feature_detected {
    ("altivec") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::altivec)
    };
    ("vsx") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::vsx)
    };
    ("power8") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::power8)
    };
    ($t:tt) => { compile_error!(concat!("unknown arm target feature: ", $t)) };
}


/// PowerPC CPU Feature enum. Each variant denotes a position in a bitset
/// for a particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum Feature {
    /// Altivec
    altivec,
    /// VSX
    vsx,
    /// Power8
    power8,
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

    // The values are part of the platform-specific [asm/cputable.h][cputable]
    //
    // [cputable]: https://github.com/torvalds/linux/blob/master/arch/powerpc/include/uapi/asm/cputable.h
    if let Ok(auxv) = linux::auxv() {
        // note: the PowerPC values are the mask to do the test (instead of the
        // index of the bit to test like in ARM and Aarch64)
        enable_feature(Feature::altivec, auxv.hwcap & 0x10000000 != 0);
        enable_feature(Feature::vsx, auxv.hwcap & 0x00000080 != 0);
        enable_feature(Feature::power8, auxv.hwcap & 0x80000000 != 0);
        return
    }

    // PowerPC's /proc/cpuinfo lacks a proper Feature field,
    // but `altivec` support is indicated in the `cpu` field.
    if let Ok(c) = linux::CpuInfo::new() {
        enable_feature(Feature::altivec, c.field("cpu").has("altivec"));
        return
    }
}
