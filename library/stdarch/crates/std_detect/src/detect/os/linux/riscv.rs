//! Run-time feature detection for RISC-V on Linux.

use super::auxvec;
use crate::detect::{Feature, bit, cache};

/// Read list of supported features from the auxiliary vector.
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let enable_feature = |value: &mut cache::Initializer, feature, enable| {
        if enable {
            value.set(feature as u32);
        }
    };
    let enable_features = |value: &mut cache::Initializer, feature_slice: &[Feature], enable| {
        if enable {
            for feature in feature_slice {
                value.set(*feature as u32);
            }
        }
    };

    // The values are part of the platform-specific [asm/hwcap.h][hwcap]
    //
    // [hwcap]: https://github.com/torvalds/linux/blob/master/arch/riscv/include/asm/hwcap.h
    let auxv = auxvec::auxv().expect("read auxvec"); // should not fail on RISC-V platform
    enable_feature(
        &mut value,
        Feature::a,
        bit::test(auxv.hwcap, (b'a' - b'a').into()),
    );
    enable_feature(
        &mut value,
        Feature::c,
        bit::test(auxv.hwcap, (b'c' - b'a').into()),
    );
    enable_features(
        &mut value,
        &[Feature::d, Feature::f, Feature::zicsr],
        bit::test(auxv.hwcap, (b'd' - b'a').into()),
    );
    enable_features(
        &mut value,
        &[Feature::f, Feature::zicsr],
        bit::test(auxv.hwcap, (b'f' - b'a').into()),
    );
    let has_i = bit::test(auxv.hwcap, (b'i' - b'a').into());
    // If future RV128I is supported, implement with `enable_feature` here
    #[cfg(target_pointer_width = "64")]
    enable_feature(&mut value, Feature::rv64i, has_i);
    #[cfg(target_pointer_width = "32")]
    enable_feature(&mut value, Feature::rv32i, has_i);
    #[cfg(target_pointer_width = "32")]
    enable_feature(
        &mut value,
        Feature::rv32e,
        bit::test(auxv.hwcap, (b'e' - b'a').into()),
    );
    enable_feature(
        &mut value,
        Feature::h,
        bit::test(auxv.hwcap, (b'h' - b'a').into()),
    );
    enable_feature(
        &mut value,
        Feature::m,
        bit::test(auxv.hwcap, (b'm' - b'a').into()),
    );
    // FIXME: Auxvec does not show supervisor feature support, but this mode may be useful
    // to detect when Rust is used to write Linux kernel modules.
    // These should be more than Auxvec way to detect supervisor features.

    value
}
