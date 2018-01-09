//! Run-time feature detection for ARM32 on Linux and `core`.

use runtime::bit;
use runtime::linux::auxv::AuxVec;
use runtime::arch::{HasFeature, __Feature};

/// Probe the ELF Auxiliary vector for hardware capabilities
///
/// The values are part of the platform-specific [asm/hwcap.h][hwcap]
///
/// [hwcap]: https://github.com/torvalds/linux/blob/master/arch/arm64/include/uapi/asm/hwcap.h
impl HasFeature for AuxVec {
    fn has_feature(&mut self, x: &__Feature) -> bool {
        use self::__Feature::*;
        match *x {
            neon => bit::test(self.hwcap, 12),
            pmull => bit::test(self.hwcap2, 1),
        }
    }
}
