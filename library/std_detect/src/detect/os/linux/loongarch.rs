//! Run-time feature detection for LoongArch on Linux.

use core::arch::asm;

use super::auxvec;
use crate::detect::{Feature, bit, cache};

/// Try to read the features from the auxiliary vector.
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let enable_feature = |value: &mut cache::Initializer, feature, enable| {
        if enable {
            value.set(feature as u32);
        }
    };

    // The values are part of the platform-specific [cpucfg]
    //
    // [cpucfg]: LoongArch Reference Manual Volume 1: Basic Architecture v1.1
    let cpucfg1: usize;
    let cpucfg2: usize;
    let cpucfg3: usize;
    unsafe {
        asm!(
            "cpucfg {}, {}",
            "cpucfg {}, {}",
            "cpucfg {}, {}",
            out(reg) cpucfg1, in(reg) 1,
            out(reg) cpucfg2, in(reg) 2,
            out(reg) cpucfg3, in(reg) 3,
            options(pure, nomem, preserves_flags, nostack)
        );
    }
    enable_feature(&mut value, Feature::_32s, bit::test(cpucfg1, 0) || bit::test(cpucfg1, 1));
    enable_feature(&mut value, Feature::frecipe, bit::test(cpucfg2, 25));
    enable_feature(&mut value, Feature::div32, bit::test(cpucfg2, 26));
    enable_feature(&mut value, Feature::lam_bh, bit::test(cpucfg2, 27));
    enable_feature(&mut value, Feature::lamcas, bit::test(cpucfg2, 28));
    enable_feature(&mut value, Feature::scq, bit::test(cpucfg2, 30));
    enable_feature(&mut value, Feature::ld_seq_sa, bit::test(cpucfg3, 23));

    // The values are part of the platform-specific [asm/hwcap.h][hwcap]
    //
    // [hwcap]: https://github.com/torvalds/linux/blob/master/arch/loongarch/include/uapi/asm/hwcap.h
    if let Ok(auxv) = auxvec::auxv() {
        enable_feature(&mut value, Feature::f, bit::test(cpucfg2, 1) && bit::test(auxv.hwcap, 3));
        enable_feature(&mut value, Feature::d, bit::test(cpucfg2, 2) && bit::test(auxv.hwcap, 3));
        enable_feature(&mut value, Feature::lsx, bit::test(auxv.hwcap, 4));
        enable_feature(&mut value, Feature::lasx, bit::test(auxv.hwcap, 5));
        enable_feature(
            &mut value,
            Feature::lbt,
            bit::test(auxv.hwcap, 10) && bit::test(auxv.hwcap, 11) && bit::test(auxv.hwcap, 12),
        );
        enable_feature(&mut value, Feature::lvz, bit::test(auxv.hwcap, 9));
        enable_feature(&mut value, Feature::ual, bit::test(auxv.hwcap, 2));
        return value;
    }
    value
}
