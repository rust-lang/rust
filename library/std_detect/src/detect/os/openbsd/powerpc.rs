//! Run-time feature detection for PowerPC on OpenBSD.
//!
//! Constants for 32-bit are located at
//! <https://github.com/openbsd/src/blob/master/sys/arch/powerpc/include/elf.h>, and for 64-bit
//! <https://github.com/openbsd/src/blob/master/sys/arch/powerpc64/include/elf.h>.

use super::auxvec;
use crate::detect::{Feature, cache};

/* Note that most constants are the same for 32- and 64-bit, but not all are */

const PPC_FEATURE_HAS_ALTIVEC: usize = 0x10000000;
const PPC_FEATURE_HAS_VSX: usize = 0x00000080;
/// ISA 2.07, typically power8
const PPC_FEATURE2_ARCH_2_07: usize = 0x80000000;

pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let enable_feature = |value: &mut cache::Initializer, f, enable| {
        if enable {
            value.set(f as u32);
        }
    };

    if let Ok(auxv) = auxvec::auxv() {
        enable_feature(&mut value, Feature::altivec, auxv.hwcap & PPC_FEATURE_HAS_ALTIVEC != 0);
        enable_feature(&mut value, Feature::vsx, auxv.hwcap & PPC_FEATURE_HAS_VSX != 0);
        enable_feature(&mut value, Feature::power8, auxv.hwcap2 & PPC_FEATURE2_ARCH_2_07 != 0);
        return value;
    }
    value
}
