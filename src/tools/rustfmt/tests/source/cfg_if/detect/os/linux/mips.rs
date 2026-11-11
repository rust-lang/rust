//! Run-time feature detection for MIPS on Linux.

use crate::detect::{Feature, cache, bit};
use super::auxvec;

/// Performs run-time feature detection.
#[inline]
pub fn check_for(x: Feature) -> bool {
    cache::test(x as u32, detect_features)
}

/// Try to read the features from the auxiliary vector, and if that fails, try
/// to read them from `/proc/cpuinfo`.
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
        enable_feature(&mut value, Feature::msa, bit::test(auxv.hwcap, 1));
        return value;
    }
    // TODO: fall back via `cpuinfo`.
    value
}
