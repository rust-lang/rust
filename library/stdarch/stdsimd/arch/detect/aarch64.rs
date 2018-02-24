//! Run-time feature detection on ARM Aarch64.

use super::bit;
use super::cache;
use super::linux;

#[macro_export]
#[unstable(feature = "stdsimd", issue = "0")]
macro_rules! is_target_feature_detected {
    ("neon") => {
        // FIXME: this should be removed once we rename Aarch64 neon to asimd
        $crate::arch::detect::check_for($crate::arch::detect::Feature::asimd)
    };
    ("asimd") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::asimd)
    };
    ("pmull") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::pmull)
    };
    ($t:tt) => { compile_error!(concat!("unknown arm target feature: ", $t)) };
}

/// ARM Aarch64 CPU Feature enum. Each variant denotes a position in a bitset
/// for a particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum Feature {
    /// ARM Advanced SIMD (ASIMD) - Aarch64
    asimd,
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
        enable_feature(Feature::asimd, bit::test(auxv.hwcap, 1));
        enable_feature(Feature::pmull, bit::test(auxv.hwcap, 4));
        return
    }

    if let Ok(c) = linux::CpuInfo::new() {
        enable_feature(Feature::asimd, c.field("Features").has("asimd"));
        enable_feature(Feature::pmull, c.field("Features").has("pmull"));
        return
    }
}
