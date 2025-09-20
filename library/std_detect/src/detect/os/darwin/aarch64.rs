//! Run-time feature detection for aarch64 on Darwin (macOS/iOS/tvOS/watchOS/visionOS).
//!
//! <https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics>

use core::ffi::CStr;

use crate::detect::{Feature, cache};

#[inline]
fn _sysctlbyname(name: &CStr) -> bool {
    use libc;

    let mut enabled: i32 = 0;
    let mut enabled_len: usize = 4;
    let enabled_ptr = &mut enabled as *mut i32 as *mut libc::c_void;

    let ret = unsafe {
        libc::sysctlbyname(name.as_ptr(), enabled_ptr, &mut enabled_len, core::ptr::null_mut(), 0)
    };

    match ret {
        0 => enabled != 0,
        _ => false,
    }
}

/// Try to read the features using sysctlbyname.
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();

    let mut enable_feature = |f, enable| {
        if enable {
            value.set(f as u32);
        }
    };

    // Armv8.0 features not using the standard identifiers
    let fp = _sysctlbyname(c"hw.optional.floatingpoint");
    let asimd = _sysctlbyname(c"hw.optional.AdvSIMD");
    let crc_old = _sysctlbyname(c"hw.optional.armv8_crc32");

    // Armv8 and Armv9 features using the standard identifiers
    let aes = _sysctlbyname(c"hw.optional.arm.FEAT_AES");
    let bf16 = _sysctlbyname(c"hw.optional.arm.FEAT_BF16");
    let bti = _sysctlbyname(c"hw.optional.arm.FEAT_BTI");
    let crc = _sysctlbyname(c"hw.optional.arm.FEAT_CRC32");
    let cssc = _sysctlbyname(c"hw.optional.arm.FEAT_CSSC");
    let dit = _sysctlbyname(c"hw.optional.arm.FEAT_DIT");
    let dotprod = _sysctlbyname(c"hw.optional.arm.FEAT_DotProd");
    let dpb = _sysctlbyname(c"hw.optional.arm.FEAT_DPB");
    let dpb2 = _sysctlbyname(c"hw.optional.arm.FEAT_DPB2");
    let ecv = _sysctlbyname(c"hw.optional.arm.FEAT_ECV");
    let fcma = _sysctlbyname(c"hw.optional.arm.FEAT_FCMA");
    let fhm = _sysctlbyname(c"hw.optional.arm.FEAT_FHM");
    let flagm = _sysctlbyname(c"hw.optional.arm.FEAT_FlagM");
    let flagm2 = _sysctlbyname(c"hw.optional.arm.FEAT_FlagM2");
    let fp16 = _sysctlbyname(c"hw.optional.arm.FEAT_FP16");
    let frintts = _sysctlbyname(c"hw.optional.arm.FEAT_FRINTTS");
    let hbc = _sysctlbyname(c"hw.optional.arm.FEAT_HBC");
    let i8mm = _sysctlbyname(c"hw.optional.arm.FEAT_I8MM");
    let jsconv = _sysctlbyname(c"hw.optional.arm.FEAT_JSCVT");
    let rcpc = _sysctlbyname(c"hw.optional.arm.FEAT_LRCPC");
    let rcpc2 = _sysctlbyname(c"hw.optional.arm.FEAT_LRCPC2");
    let lse = _sysctlbyname(c"hw.optional.arm.FEAT_LSE");
    let lse2 = _sysctlbyname(c"hw.optional.arm.FEAT_LSE2");
    let mte = _sysctlbyname(c"hw.optional.arm.FEAT_MTE");
    let mte2 = _sysctlbyname(c"hw.optional.arm.FEAT_MTE2");
    let pauth = _sysctlbyname(c"hw.optional.arm.FEAT_PAuth");
    let pmull = _sysctlbyname(c"hw.optional.arm.FEAT_PMULL");
    let rdm = _sysctlbyname(c"hw.optional.arm.FEAT_RDM");
    let sb = _sysctlbyname(c"hw.optional.arm.FEAT_SB");
    let sha1 = _sysctlbyname(c"hw.optional.arm.FEAT_SHA1");
    let sha256 = _sysctlbyname(c"hw.optional.arm.FEAT_SHA256");
    let sha3 = _sysctlbyname(c"hw.optional.arm.FEAT_SHA3");
    let sha512 = _sysctlbyname(c"hw.optional.arm.FEAT_SHA512");
    let sme = _sysctlbyname(c"hw.optional.arm.FEAT_SME");
    let sme2 = _sysctlbyname(c"hw.optional.arm.FEAT_SME2");
    let sme2p1 = _sysctlbyname(c"hw.optional.arm.FEAT_SME2p1");
    let sme_f64f64 = _sysctlbyname(c"hw.optional.arm.FEAT_SME_F64F64");
    let sme_i16i64 = _sysctlbyname(c"hw.optional.arm.FEAT_SME_I16I64");
    let ssbs = _sysctlbyname(c"hw.optional.arm.FEAT_SSBS");
    let wfxt = _sysctlbyname(c"hw.optional.arm.FEAT_WFxT");

    // The following features are not exposed by `is_aarch64_feature_detected`,
    // but *are* reported by `sysctl`. They are here as documentation that they
    // exist, and may potentially be exposed later.
    /*
    let afp = _sysctlbyname(c"hw.optional.arm.FEAT_AFP");
    let csv2 = _sysctlbyname(c"hw.optional.arm.FEAT_CSV2");
    let csv3 = _sysctlbyname(c"hw.optional.arm.FEAT_CSV3");
    let ebf16 = _sysctlbyname(c"hw.optional.arm.FEAT_EBF16");
    let fpac = _sysctlbyname(c"hw.optional.arm.FEAT_FPAC");
    let fpaccombine = _sysctlbyname(c"hw.optional.arm.FEAT_FPACCOMBINE");
    let mte_async = _sysctlbyname(c"hw.optional.arm.FEAT_MTE_ASYNC");
    let mte_canonical_tags = _sysctlbyname(c"hw.optional.arm.FEAT_MTE_CANONICAL_TAGS");
    let mte_no_address_tags = _sysctlbyname(c"hw.optional.arm.FEAT_MTE_NO_ADDRESS_TAGS");
    let mte_store_only = _sysctlbyname(c"hw.optional.arm.FEAT_MTE_STORE_ONLY");
    let mte3 = _sysctlbyname(c"hw.optional.arm.FEAT_MTE3");
    let mte4 = _sysctlbyname(c"hw.optional.arm.FEAT_MTE4");
    let pacimp = _sysctlbyname(c"hw.optional.arm.FEAT_PACIMP");
    let pauth2 = _sysctlbyname(c"hw.optional.arm.FEAT_PAuth2");
    let rpres = _sysctlbyname(c"hw.optional.arm.FEAT_RPRES");
    let specres = _sysctlbyname(c"hw.optional.arm.FEAT_SPECRES");
    let specres2 = _sysctlbyname(c"hw.optional.arm.FEAT_SPECRES2");
     */

    // The following "features" are reported by `sysctl` but are mandatory parts
    // of SME or SME2, and so are not exposed separately by
    // `is_aarch64_feature_detected`.  They are here to document their
    // existence, in case they're needed in the future.
    /*
    let sme_b16f32 = _sysctlbyname(c"hw.optional.arm.SME_B16F32");
    let sme_bi32i32 = _sysctlbyname(c"hw.optional.arm.SME_BI32I32");
    let sme_f16f32 = _sysctlbyname(c"hw.optional.arm.SME_F16F32");
    let sme_f32f32 = _sysctlbyname(c"hw.optional.arm.SME_F32F32");
    let sme_i16i32 = _sysctlbyname(c"hw.optional.arm.SME_I16I32");
    let sme_i8i32 = _sysctlbyname(c"hw.optional.arm.SME_I8I32");
     */

    enable_feature(Feature::aes, aes && pmull);
    enable_feature(Feature::asimd, asimd);
    enable_feature(Feature::bf16, bf16);
    enable_feature(Feature::bti, bti);
    enable_feature(Feature::crc, crc_old || crc);
    enable_feature(Feature::cssc, cssc);
    enable_feature(Feature::dit, dit);
    enable_feature(Feature::dotprod, dotprod);
    enable_feature(Feature::dpb, dpb);
    enable_feature(Feature::dpb2, dpb2);
    enable_feature(Feature::ecv, ecv);
    enable_feature(Feature::fcma, fcma);
    enable_feature(Feature::fhm, fhm);
    enable_feature(Feature::flagm, flagm);
    enable_feature(Feature::flagm2, flagm2);
    enable_feature(Feature::fp, fp);
    enable_feature(Feature::fp16, fp16);
    enable_feature(Feature::frintts, frintts);
    enable_feature(Feature::hbc, hbc);
    enable_feature(Feature::i8mm, i8mm);
    enable_feature(Feature::jsconv, jsconv);
    enable_feature(Feature::lse, lse);
    enable_feature(Feature::lse2, lse2);
    enable_feature(Feature::mte, mte && mte2);
    enable_feature(Feature::paca, pauth);
    enable_feature(Feature::pacg, pauth);
    enable_feature(Feature::pmull, aes && pmull);
    enable_feature(Feature::rcpc, rcpc);
    enable_feature(Feature::rcpc2, rcpc2);
    enable_feature(Feature::rdm, rdm);
    enable_feature(Feature::sb, sb);
    enable_feature(Feature::sha2, sha1 && sha256 && asimd);
    enable_feature(Feature::sha3, sha512 && sha3 && asimd);
    enable_feature(Feature::sme, sme);
    enable_feature(Feature::sme2, sme2);
    enable_feature(Feature::sme2p1, sme2p1);
    enable_feature(Feature::sme_f64f64, sme_f64f64);
    enable_feature(Feature::sme_i16i64, sme_i16i64);
    enable_feature(Feature::ssbs, ssbs);
    enable_feature(Feature::wfxt, wfxt);

    value
}
