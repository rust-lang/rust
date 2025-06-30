//! Run-time feature detection for Aarch64 on Windows.

use crate::detect::{Feature, cache};

/// Try to read the features using IsProcessorFeaturePresent.
pub(crate) fn detect_features() -> cache::Initializer {
    type DWORD = u32;
    type BOOL = i32;

    const FALSE: BOOL = 0;
    // The following Microsoft documents isn't updated for aarch64.
    // https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent
    // These are defined in winnt.h of Windows SDK
    const PF_ARM_VFP_32_REGISTERS_AVAILABLE: u32 = 18;
    const PF_ARM_NEON_INSTRUCTIONS_AVAILABLE: u32 = 19;
    const PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE: u32 = 30;
    const PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE: u32 = 31;
    const PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE: u32 = 34;
    const PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE: u32 = 43;
    const PF_ARM_V83_JSCVT_INSTRUCTIONS_AVAILABLE: u32 = 44;
    const PF_ARM_V83_LRCPC_INSTRUCTIONS_AVAILABLE: u32 = 45;
    const PF_ARM_SVE_INSTRUCTIONS_AVAILABLE: u32 = 46;
    const PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE: u32 = 47;
    const PF_ARM_SVE2_1_INSTRUCTIONS_AVAILABLE: u32 = 48;
    const PF_ARM_SVE_AES_INSTRUCTIONS_AVAILABLE: u32 = 49;
    const PF_ARM_SVE_PMULL128_INSTRUCTIONS_AVAILABLE: u32 = 50;
    const PF_ARM_SVE_BITPERM_INSTRUCTIONS_AVAILABLE: u32 = 51;
    // const PF_ARM_SVE_BF16_INSTRUCTIONS_AVAILABLE: u32 = 52;
    // const PF_ARM_SVE_EBF16_INSTRUCTIONS_AVAILABLE: u32 = 53;
    const PF_ARM_SVE_B16B16_INSTRUCTIONS_AVAILABLE: u32 = 54;
    const PF_ARM_SVE_SHA3_INSTRUCTIONS_AVAILABLE: u32 = 55;
    const PF_ARM_SVE_SM4_INSTRUCTIONS_AVAILABLE: u32 = 56;
    // const PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE: u32 = 57;
    // const PF_ARM_SVE_F32MM_INSTRUCTIONS_AVAILABLE: u32 = 58;
    // const PF_ARM_SVE_F64MM_INSTRUCTIONS_AVAILABLE: u32 = 59;

    unsafe extern "system" {
        fn IsProcessorFeaturePresent(ProcessorFeature: DWORD) -> BOOL;
    }

    let mut value = cache::Initializer::default();
    {
        let mut enable_feature = |f, enable| {
            if enable {
                value.set(f as u32);
            }
        };

        // Some features may be supported on current CPU,
        // but no way to detect it by OS API.
        // Also, we require unsafe block for the extern "system" calls.
        unsafe {
            enable_feature(
                Feature::fp,
                IsProcessorFeaturePresent(PF_ARM_VFP_32_REGISTERS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::asimd,
                IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::crc,
                IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::lse,
                IsProcessorFeaturePresent(PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::dotprod,
                IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::jsconv,
                IsProcessorFeaturePresent(PF_ARM_V83_JSCVT_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::rcpc,
                IsProcessorFeaturePresent(PF_ARM_V83_LRCPC_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::sve,
                IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::sve2,
                IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::sve2p1,
                IsProcessorFeaturePresent(PF_ARM_SVE2_1_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::sve2_aes,
                IsProcessorFeaturePresent(PF_ARM_SVE_AES_INSTRUCTIONS_AVAILABLE) != FALSE
                    && IsProcessorFeaturePresent(PF_ARM_SVE_PMULL128_INSTRUCTIONS_AVAILABLE)
                        != FALSE,
            );
            enable_feature(
                Feature::sve2_bitperm,
                IsProcessorFeaturePresent(PF_ARM_SVE_BITPERM_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::sve_b16b16,
                IsProcessorFeaturePresent(PF_ARM_SVE_B16B16_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::sve2_sha3,
                IsProcessorFeaturePresent(PF_ARM_SVE_SHA3_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::sve2_sm4,
                IsProcessorFeaturePresent(PF_ARM_SVE_SM4_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            // PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE means aes, sha1, sha2 and
            // pmull support
            let crypto =
                IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE) != FALSE;
            enable_feature(Feature::aes, crypto);
            enable_feature(Feature::pmull, crypto);
            enable_feature(Feature::sha2, crypto);
        }
    }
    value
}
