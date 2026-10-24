//! Run-time feature detection for ARM on FreeBSD

use super::auxvec;
use crate::detect::{Feature, cache};

// Defined in machine/elf.h.
// https://github.com/freebsd/freebsd-src/blob/deb63adf945d446ed91a9d84124c71f15ae571d1/sys/arm/include/elf.h
const HWCAP_NEON: usize = 0x00001000;
const HWCAP2_AES: usize = 0x00000001;
const HWCAP2_PMULL: usize = 0x00000002;
const HWCAP2_SHA1: usize = 0x00000004;
const HWCAP2_SHA2: usize = 0x00000008;
const HWCAP2_CRC32: usize = 0x00000010;

/// Try to read the features from the auxiliary vector
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let enable_feature = |value: &mut cache::Initializer, f, enable| {
        if enable {
            value.set(f as u32);
        }
    };

    if let Ok(auxv) = auxvec::auxv() {
        enable_feature(&mut value, Feature::neon, auxv.hwcap & HWCAP_NEON != 0);
        enable_feature(&mut value, Feature::pmull, auxv.hwcap2 & HWCAP2_PMULL != 0);
        enable_feature(&mut value, Feature::crc, auxv.hwcap2 & HWCAP2_CRC32 != 0);
        enable_feature(&mut value, Feature::aes, auxv.hwcap2 & HWCAP2_AES != 0);
        // SHA2 requires SHA1 & SHA2 features
        let sha1 = auxv.hwcap2 & HWCAP2_SHA1 != 0;
        let sha2 = auxv.hwcap2 & HWCAP2_SHA2 != 0;
        enable_feature(&mut value, Feature::sha2, sha1 && sha2);
        return value;
    }
    value
}
