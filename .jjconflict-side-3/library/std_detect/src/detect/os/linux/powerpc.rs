//! Run-time feature detection for PowerPC on Linux.

use super::auxvec;
use crate::detect::{Feature, cache};

/// Try to read the features from the auxiliary vector.
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let enable_feature = |value: &mut cache::Initializer, f, enable| {
        if enable {
            value.set(f as u32);
        }
    };

    // The values are part of the platform-specific [asm/cputable.h][cputable]
    //
    // [cputable]: https://github.com/torvalds/linux/blob/master/arch/powerpc/include/uapi/asm/cputable.h
    if let Ok(auxv) = auxvec::auxv() {
        // note: the PowerPC values are the mask to do the test (instead of the
        // index of the bit to test like in ARM and Aarch64)
        enable_feature(&mut value, Feature::altivec, auxv.hwcap & 0x10000000 != 0);
        enable_feature(&mut value, Feature::vsx, auxv.hwcap & 0x00000080 != 0);
        let power8_features = auxv.hwcap2 & 0x80000000 != 0;
        enable_feature(&mut value, Feature::power8, power8_features);
        enable_feature(&mut value, Feature::power8_altivec, power8_features);
        enable_feature(&mut value, Feature::power8_crypto, power8_features);
        enable_feature(&mut value, Feature::power8_vector, power8_features);
        let power9_features = auxv.hwcap2 & 0x00800000 != 0;
        enable_feature(&mut value, Feature::power9, power9_features);
        enable_feature(&mut value, Feature::power9_altivec, power9_features);
        enable_feature(&mut value, Feature::power9_vector, power9_features);
        return value;
    }
    value
}
