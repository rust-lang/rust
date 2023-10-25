//! Run-time feature detection for LoongArch on Linux.

use super::auxvec;
use crate::detect::{bit, cache, Feature};

/// Try to read the features from the auxiliary vector.
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let enable_feature = |value: &mut cache::Initializer, feature, enable| {
        if enable {
            value.set(feature as u32);
        }
    };

    // The values are part of the platform-specific [asm/hwcap.h][hwcap]
    //
    // [hwcap]: https://github.com/torvalds/linux/blob/master/arch/loongarch/include/uapi/asm/hwcap.h
    if let Ok(auxv) = auxvec::auxv() {
        enable_feature(&mut value, Feature::lam, bit::test(auxv.hwcap, 1));
        enable_feature(&mut value, Feature::ual, bit::test(auxv.hwcap, 2));
        enable_feature(&mut value, Feature::fpu, bit::test(auxv.hwcap, 3));
        enable_feature(&mut value, Feature::lsx, bit::test(auxv.hwcap, 4));
        enable_feature(&mut value, Feature::lasx, bit::test(auxv.hwcap, 5));
        enable_feature(&mut value, Feature::crc32, bit::test(auxv.hwcap, 6));
        enable_feature(&mut value, Feature::complex, bit::test(auxv.hwcap, 7));
        enable_feature(&mut value, Feature::crypto, bit::test(auxv.hwcap, 8));
        enable_feature(&mut value, Feature::lvz, bit::test(auxv.hwcap, 9));
        enable_feature(&mut value, Feature::lbtx86, bit::test(auxv.hwcap, 10));
        enable_feature(&mut value, Feature::lbtarm, bit::test(auxv.hwcap, 11));
        enable_feature(&mut value, Feature::lbtmips, bit::test(auxv.hwcap, 12));
        return value;
    }
    value
}
