//! Run-time feature detection for ARM on Linux.

use super::auxvec;
use crate::detect::{Feature, bit, cache};

/// Try to read the features from the auxiliary vector.
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let enable_feature = |value: &mut cache::Initializer, f, enable| {
        if enable {
            value.set(f as u32);
        }
    };

    // The values are part of the platform-specific [asm/hwcap.h][hwcap]
    //
    // [hwcap]: https://github.com/torvalds/linux/blob/master/arch/arm/include/uapi/asm/hwcap.h
    if let Ok(auxv) = auxvec::auxv() {
        enable_feature(&mut value, Feature::i8mm, bit::test(auxv.hwcap, 27));
        enable_feature(&mut value, Feature::dotprod, bit::test(auxv.hwcap, 24));
        enable_feature(&mut value, Feature::neon, bit::test(auxv.hwcap, 12));
        enable_feature(&mut value, Feature::pmull, bit::test(auxv.hwcap2, 1));
        enable_feature(&mut value, Feature::crc, bit::test(auxv.hwcap2, 4));
        enable_feature(&mut value, Feature::aes, bit::test(auxv.hwcap2, 0));
        // SHA2 requires SHA1 & SHA2 features
        enable_feature(
            &mut value,
            Feature::sha2,
            bit::test(auxv.hwcap2, 2) && bit::test(auxv.hwcap2, 3),
        );
        return value;
    }
    value
}
