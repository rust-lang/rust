//! Run-time feature detection for Aarch64 on Windows.

use crate::detect::{cache, Feature};

/// Try to read the features using IsProcessorFeaturePresent.
pub(crate) fn detect_features() -> cache::Initializer {
    type DWORD = u32;
    type BOOL = i32;

    const FALSE: BOOL = 0;
    // The following Microsoft documents isn't updated for aarch64.
    // https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent
    // These are defined in winnt.h of Windows SDK
    const PF_ARM_NEON_INSTRUCTIONS_AVAILABLE: u32 = 19;
    const PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE: u32 = 30;
    const PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE: u32 = 31;

    extern "system" {
        pub fn IsProcessorFeaturePresent(ProcessorFeature: DWORD) -> BOOL;
    }

    let mut value = cache::Initializer::default();
    {
        let mut enable_feature = |f, enable| {
            if enable {
                value.set(f as u32);
            }
        };

        // Some features such Feature::fp may be supported on current CPU,
        // but no way to detect it by OS API.
        // Also, we require unsafe block for the extern "system" calls.
        unsafe {
            enable_feature(
                Feature::asimd,
                IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::crc,
                IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            // PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE means aes, sha1, sha2 and
            // pmull support
            enable_feature(
                Feature::aes,
                IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::pmull,
                IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::sha2,
                IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
        }
    }
    value
}
