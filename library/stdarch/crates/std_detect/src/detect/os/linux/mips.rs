//! Run-time feature detection for MIPS on Linux.

use super::auxvec;
use crate::detect::{Feature, bit, cache};

/// Try to read the features from the auxiliary vector, and if that fails, try
/// to read them from `/proc/cpuinfo`.
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let enable_feature = |value: &mut cache::Initializer, f, enable| {
        if enable {
            value.set(f as u32);
        }
    };

    // The values are part of the platform-specific [asm/hwcap.h][hwcap]
    //
    // [hwcap]: https://github.com/torvalds/linux/blob/master/arch/mips/include/uapi/asm/hwcap.h
    if let Ok(auxv) = auxvec::auxv() {
        enable_feature(&mut value, Feature::msa, bit::test(auxv.hwcap, 1));
        return value;
    }
    // TODO: fall back via `cpuinfo`.
    value
}
