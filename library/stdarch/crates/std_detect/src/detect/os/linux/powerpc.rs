//! Run-time feature detection for PowerPC on Linux.

use super::auxvec;
use crate::detect::{cache, Feature};

/// Try to read the features from the auxiliary vector, and if that fails, try
/// to read them from /proc/cpuinfo.
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
        enable_feature(&mut value, Feature::power8, auxv.hwcap2 & 0x80000000 != 0);
        return value;
    }

    // PowerPC's /proc/cpuinfo lacks a proper Feature field,
    // but `altivec` support is indicated in the `cpu` field.
    #[cfg(feature = "std_detect_file_io")]
    if let Ok(c) = super::cpuinfo::CpuInfo::new() {
        enable_feature(&mut value, Feature::altivec, c.field("cpu").has("altivec"));
        return value;
    }
    value
}
