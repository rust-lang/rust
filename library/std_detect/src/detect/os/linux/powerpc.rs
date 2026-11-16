//! Run-time feature detection for PowerPC on Linux.
//!
//! Constants for both 32- and 64-bit come from
//! <https://github.com/torvalds/linux/blob/master/arch/powerpc/include/uapi/asm/cputable.h>.

use super::auxvec;
use crate::detect::{Feature, cache};

const PPC_FEATURE_HAS_ALTIVEC: usize = 0x10000000;
const PPC_FEATURE_HAS_VSX: usize = 0x00000080;
/// ISA 2.07, typically power8
const PPC_FEATURE2_ARCH_2_07: usize = 0x80000000;
/// ISA 3.00, typically power9
const PPC_FEATURE2_ARCH_3_00: usize = 0x00800000;

/// Try to read the features from the auxiliary vector.
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let enable_feature = |value: &mut cache::Initializer, f, enable| {
        if enable {
            value.set(f as u32);
        }
    };

    if let Ok(auxv) = auxvec::auxv() {
        // note: the PowerPC values are the mask to do the test (instead of the
        // index of the bit to test like in ARM and Aarch64)
        enable_feature(&mut value, Feature::altivec, auxv.hwcap & PPC_FEATURE_HAS_ALTIVEC != 0);
        enable_feature(&mut value, Feature::vsx, auxv.hwcap & PPC_FEATURE_HAS_VSX != 0);
        // FIXME(powerpc_target_feature): this code assumes the ISA hwcaps imply vsx/altivec, but
        // this is not a correct assumption. See rust-lang/compiler-team#1046 for
        // a possible resolution.
        let power8_features = auxv.hwcap2 & PPC_FEATURE2_ARCH_2_07 != 0;
        enable_feature(&mut value, Feature::power8, power8_features);
        enable_feature(&mut value, Feature::power8_altivec, power8_features);
        enable_feature(&mut value, Feature::power8_crypto, power8_features);
        enable_feature(&mut value, Feature::power8_vector, power8_features);
        let power9_features = auxv.hwcap2 & PPC_FEATURE2_ARCH_3_00 != 0;
        enable_feature(&mut value, Feature::power9, power9_features);
        enable_feature(&mut value, Feature::power9_altivec, power9_features);
        enable_feature(&mut value, Feature::power9_vector, power9_features);
        return value;
    }
    value
}
