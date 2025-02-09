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

    unsafe extern "system" {
        pub fn IsProcessorFeaturePresent(ProcessorFeature: DWORD) -> BOOL;
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
