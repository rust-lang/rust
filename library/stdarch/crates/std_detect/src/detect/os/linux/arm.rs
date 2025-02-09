//! Run-time feature detection for ARM on Linux.

use super::auxvec;
use crate::detect::{Feature, bit, cache};

/// Try to read the features from the auxiliary vector, and if that fails, try
/// to read them from /proc/cpuinfo.
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

    #[cfg(feature = "std_detect_file_io")]
    if let Ok(c) = super::cpuinfo::CpuInfo::new() {
        enable_feature(
            &mut value,
            Feature::neon,
            c.field("Features").has("neon") && !has_broken_neon(&c),
        );
        enable_feature(&mut value, Feature::i8mm, c.field("Features").has("i8mm"));
        enable_feature(
            &mut value,
            Feature::dotprod,
            c.field("Features").has("asimddp"),
        );
        enable_feature(&mut value, Feature::pmull, c.field("Features").has("pmull"));
        enable_feature(&mut value, Feature::crc, c.field("Features").has("crc32"));
        enable_feature(&mut value, Feature::aes, c.field("Features").has("aes"));
        enable_feature(
            &mut value,
            Feature::sha2,
            c.field("Features").has("sha1") && c.field("Features").has("sha2"),
        );
        return value;
    }
    value
}

/// Is the CPU known to have a broken NEON unit?
///
/// See https://crbug.com/341598.
#[cfg(feature = "std_detect_file_io")]
fn has_broken_neon(cpuinfo: &super::cpuinfo::CpuInfo) -> bool {
    cpuinfo.field("CPU implementer") == "0x51"
        && cpuinfo.field("CPU architecture") == "7"
        && cpuinfo.field("CPU variant") == "0x1"
        && cpuinfo.field("CPU part") == "0x04d"
        && cpuinfo.field("CPU revision") == "0"
}
