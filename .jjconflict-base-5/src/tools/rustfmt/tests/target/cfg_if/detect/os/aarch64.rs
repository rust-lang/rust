//! Run-time feature detection for Aarch64 on any OS that emulates the mrs instruction.
//!
//! On FreeBSD >= 12.0, Linux >= 4.11 and other operating systems, it is possible to use
//! privileged system registers from userspace to check CPU feature support.
//!
//! AArch64 system registers ID_AA64ISAR0_EL1, ID_AA64PFR0_EL1, ID_AA64ISAR1_EL1
//! have bits dedicated to features like AdvSIMD, CRC32, AES, atomics (LSE), etc.
//! Each part of the register indicates the level of support for a certain feature, e.g.
//! when ID_AA64ISAR0_EL1\[7:4\] is >= 1, AES is supported; when it's >= 2, PMULL is supported.
//!
//! For proper support of [SoCs where different cores have different capabilities](https://medium.com/@jadr2ddude/a-big-little-problem-a-tale-of-big-little-gone-wrong-e7778ce744bb),
//! the OS has to always report only the features supported by all cores, like [FreeBSD does](https://reviews.freebsd.org/D17137#393947).
//!
//! References:
//!
//! - [Zircon implementation](https://fuchsia.googlesource.com/zircon/+/master/kernel/arch/arm64/feature.cpp)
//! - [Linux documentation](https://www.kernel.org/doc/Documentation/arm64/cpu-feature-registers.txt)

use crate::detect::{cache, Feature};

/// Try to read the features from the system registers.
///
/// This will cause SIGILL if the current OS is not trapping the mrs instruction.
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();

    {
        let mut enable_feature = |f, enable| {
            if enable {
                value.set(f as u32);
            }
        };

        // ID_AA64ISAR0_EL1 - Instruction Set Attribute Register 0
        let aa64isar0: u64;
        unsafe {
            asm!("mrs $0, ID_AA64ISAR0_EL1" : "=r"(aa64isar0));
        }

        let aes = bits_shift(aa64isar0, 7, 4) >= 1;
        let pmull = bits_shift(aa64isar0, 7, 4) >= 2;
        let sha1 = bits_shift(aa64isar0, 11, 8) >= 1;
        let sha2 = bits_shift(aa64isar0, 15, 12) >= 1;
        enable_feature(Feature::pmull, pmull);
        // Crypto is specified as AES + PMULL + SHA1 + SHA2 per LLVM/hosts.cpp
        enable_feature(Feature::crypto, aes && pmull && sha1 && sha2);
        enable_feature(Feature::lse, bits_shift(aa64isar0, 23, 20) >= 1);
        enable_feature(Feature::crc, bits_shift(aa64isar0, 19, 16) >= 1);

        // ID_AA64PFR0_EL1 - Processor Feature Register 0
        let aa64pfr0: u64;
        unsafe {
            asm!("mrs $0, ID_AA64PFR0_EL1" : "=r"(aa64pfr0));
        }

        let fp = bits_shift(aa64pfr0, 19, 16) < 0xF;
        let fphp = bits_shift(aa64pfr0, 19, 16) >= 1;
        let asimd = bits_shift(aa64pfr0, 23, 20) < 0xF;
        let asimdhp = bits_shift(aa64pfr0, 23, 20) >= 1;
        enable_feature(Feature::fp, fp);
        enable_feature(Feature::fp16, fphp);
        // SIMD support requires float support - if half-floats are
        // supported, it also requires half-float support:
        enable_feature(Feature::asimd, fp && asimd && (!fphp | asimdhp));
        // SIMD extensions require SIMD support:
        enable_feature(Feature::rdm, asimd && bits_shift(aa64isar0, 31, 28) >= 1);
        enable_feature(
            Feature::dotprod,
            asimd && bits_shift(aa64isar0, 47, 44) >= 1,
        );
        enable_feature(Feature::sve, asimd && bits_shift(aa64pfr0, 35, 32) >= 1);

        // ID_AA64ISAR1_EL1 - Instruction Set Attribute Register 1
        let aa64isar1: u64;
        unsafe {
            asm!("mrs $0, ID_AA64ISAR1_EL1" : "=r"(aa64isar1));
        }

        enable_feature(Feature::rcpc, bits_shift(aa64isar1, 23, 20) >= 1);
    }

    value
}

#[inline]
fn bits_shift(x: u64, high: usize, low: usize) -> u64 {
    (x >> low) & ((1 << (high - low + 1)) - 1)
}
