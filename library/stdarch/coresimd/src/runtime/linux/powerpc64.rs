//! Run-time feature detection for PowerPC64 on Linux and `core`.

use runtime::linux::auxv::AuxVec;
use runtime::arch::{HasFeature, __Feature};

/// Probe the ELF Auxiliary vector for hardware capabilities
///
/// The values are part of the platform-specific [asm/cputable.h][cputable]
///
/// [cputable]: https://github.com/torvalds/linux/blob/master/arch/powerpc/include/uapi/asm/cputable.h
impl HasFeature for AuxVec {
    fn has_feature(&mut self, x: &__Feature) -> bool {
        use self::__Feature::*;
        // note: the PowerPC values are the mask to do the test (instead of the
        // index of the bit to test like in ARM and Aarch64)
        match *x {
            altivec => self.hwcap & 0x10000000 != 0,
            vsx => self.hwcap & 0x00000080 != 0,
            power8 => self.hwcap2 & 0x80000000 != 0,
        }
    }
}
