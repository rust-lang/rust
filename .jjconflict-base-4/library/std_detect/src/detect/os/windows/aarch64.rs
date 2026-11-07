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
    const PF_ARM_SVE_F32MM_INSTRUCTIONS_AVAILABLE: u32 = 58;
    const PF_ARM_SVE_F64MM_INSTRUCTIONS_AVAILABLE: u32 = 59;
    const PF_ARM_LSE2_AVAILABLE: u32 = 62;
    const PF_ARM_SHA3_INSTRUCTIONS_AVAILABLE: u32 = 64;
    const PF_ARM_SHA512_INSTRUCTIONS_AVAILABLE: u32 = 65;
    const PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE: u32 = 66;
    const PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE: u32 = 67;
    const PF_ARM_V86_BF16_INSTRUCTIONS_AVAILABLE: u32 = 68;

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

        // Some features may be supported on the current CPU but have no
        // detection path through the Win32 API; those report `false`.
        // SAFETY: `IsProcessorFeaturePresent` is a Win32 entry point taking a
        // `DWORD` by value and returning a `BOOL`. No pointer parameters,
        // no out-parameters, no thread-safety constraints.
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
            enable_feature(
                Feature::f32mm,
                IsProcessorFeaturePresent(PF_ARM_SVE_F32MM_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::f64mm,
                IsProcessorFeaturePresent(PF_ARM_SVE_F64MM_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::lse2,
                IsProcessorFeaturePresent(PF_ARM_LSE2_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::fp16,
                IsProcessorFeaturePresent(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::i8mm,
                IsProcessorFeaturePresent(PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            enable_feature(
                Feature::bf16,
                IsProcessorFeaturePresent(PF_ARM_V86_BF16_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            // stdarch `sha3` is FEAT_SHA3 + FEAT_SHA512 together; Windows
            // exposes them as two separate flags.
            enable_feature(
                Feature::sha3,
                IsProcessorFeaturePresent(PF_ARM_SHA3_INSTRUCTIONS_AVAILABLE) != FALSE
                    && IsProcessorFeaturePresent(PF_ARM_SHA512_INSTRUCTIONS_AVAILABLE) != FALSE,
            );
            // No PF_ARM_RDM_* constant exists. Derive FEAT_RDM from FEAT_DotProd:
            // DotProd is an optional v8.2-A feature only present on cores that
            // implement at least v8.1-A; v8.1-A with AdvSIMD mandates FEAT_RDM
            // (Arm ARM K.a §D17.2.91), and AdvSIMD is universal on Windows-on-ARM.
            // Same inference shipped in .NET 10 (dotnet/runtime PR 109493).
            enable_feature(
                Feature::rdm,
                IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != FALSE,
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
