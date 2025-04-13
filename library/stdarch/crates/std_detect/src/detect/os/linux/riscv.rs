//! Run-time feature detection for RISC-V on Linux.

use super::super::riscv::imply_features;
use super::auxvec;
use crate::detect::{Feature, bit, cache};

/// Read list of supported features from the auxiliary vector.
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let mut enable_feature = |feature, enable| {
        if enable {
            value.set(feature as u32);
        }
    };

    // Use auxiliary vector to enable single-letter ISA extensions.
    // The values are part of the platform-specific [asm/hwcap.h][hwcap]
    //
    // [hwcap]: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/arch/riscv/include/uapi/asm/hwcap.h?h=v6.14
    let auxv = auxvec::auxv().expect("read auxvec"); // should not fail on RISC-V platform
    #[allow(clippy::eq_op)]
    enable_feature(Feature::a, bit::test(auxv.hwcap, (b'a' - b'a').into()));
    enable_feature(Feature::c, bit::test(auxv.hwcap, (b'c' - b'a').into()));
    enable_feature(Feature::d, bit::test(auxv.hwcap, (b'd' - b'a').into()));
    enable_feature(Feature::f, bit::test(auxv.hwcap, (b'f' - b'a').into()));
    enable_feature(Feature::h, bit::test(auxv.hwcap, (b'h' - b'a').into()));
    enable_feature(Feature::m, bit::test(auxv.hwcap, (b'm' - b'a').into()));

    // Handle base ISA.
    let has_i = bit::test(auxv.hwcap, (b'i' - b'a').into());
    // If future RV128I is supported, implement with `enable_feature` here.
    // Note that we should use `target_arch` instead of `target_pointer_width`
    // to avoid misdetection caused by experimental ABIs such as RV64ILP32.
    #[cfg(target_arch = "riscv64")]
    enable_feature(Feature::rv64i, has_i);
    #[cfg(target_arch = "riscv32")]
    enable_feature(Feature::rv32i, has_i);
    // FIXME: e is not exposed in any of asm/hwcap.h, uapi/asm/hwcap.h, uapi/asm/hwprobe.h
    #[cfg(target_arch = "riscv32")]
    enable_feature(Feature::rv32e, bit::test(auxv.hwcap, (b'e' - b'a').into()));

    // FIXME: Auxvec does not show supervisor feature support, but this mode may be useful
    // to detect when Rust is used to write Linux kernel modules.
    // These should be more than Auxvec way to detect supervisor features.

    imply_features(value)
}
