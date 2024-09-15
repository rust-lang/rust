//! Run-time feature detection for aarch64 on Darwin (macOS/iOS/tvOS/watchOS/visionOS).
//!
//! <https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics>

use crate::detect::{cache, Feature};
use core::ffi::CStr;

#[inline]
fn _sysctlbyname(name: &CStr) -> bool {
    use libc;

    let mut enabled: i32 = 0;
    let mut enabled_len: usize = 4;
    let enabled_ptr = &mut enabled as *mut i32 as *mut libc::c_void;

    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            enabled_ptr,
            &mut enabled_len,
            core::ptr::null_mut(),
            0,
        )
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

    let asimd = _sysctlbyname(c"hw.optional.AdvSIMD");
    let pmull = _sysctlbyname(c"hw.optional.arm.FEAT_PMULL");
    let fp = _sysctlbyname(c"hw.optional.floatingpoint");
    let fp16 = _sysctlbyname(c"hw.optional.arm.FEAT_FP16");
    let crc = _sysctlbyname(c"hw.optional.armv8_crc32");
    let lse = _sysctlbyname(c"hw.optional.arm.FEAT_LSE");
    let lse2 = _sysctlbyname(c"hw.optional.arm.FEAT_LSE2");
    let rdm = _sysctlbyname(c"hw.optional.arm.FEAT_RDM");
    let rcpc = _sysctlbyname(c"hw.optional.arm.FEAT_LRCPC");
    let rcpc2 = _sysctlbyname(c"hw.optional.arm.FEAT_LRCPC2");
    let dotprod = _sysctlbyname(c"hw.optional.arm.FEAT_DotProd");
    let fhm = _sysctlbyname(c"hw.optional.arm.FEAT_FHM");
    let flagm = _sysctlbyname(c"hw.optional.arm.FEAT_FlagM");
    let ssbs = _sysctlbyname(c"hw.optional.arm.FEAT_SSBS");
    let sb = _sysctlbyname(c"hw.optional.arm.FEAT_SB");
    let paca = _sysctlbyname(c"hw.optional.arm.FEAT_PAuth");
    let dpb = _sysctlbyname(c"hw.optional.arm.FEAT_DPB");
    let dpb2 = _sysctlbyname(c"hw.optional.arm.FEAT_DPB2");
    let frintts = _sysctlbyname(c"hw.optional.arm.FEAT_FRINTTS");
    let i8mm = _sysctlbyname(c"hw.optional.arm.FEAT_I8MM");
    let bf16 = _sysctlbyname(c"hw.optional.arm.FEAT_BF16");
    let bti = _sysctlbyname(c"hw.optional.arm.FEAT_BTI");
    let fcma = _sysctlbyname(c"hw.optional.arm.FEAT_FCMA");
    let aes = _sysctlbyname(c"hw.optional.arm.FEAT_AES");
    let sha1 = _sysctlbyname(c"hw.optional.arm.FEAT_SHA1");
    let sha2 = _sysctlbyname(c"hw.optional.arm.FEAT_SHA256");
    let sha3 = _sysctlbyname(c"hw.optional.arm.FEAT_SHA3");
    let sha512 = _sysctlbyname(c"hw.optional.arm.FEAT_SHA512");
    let jsconv = _sysctlbyname(c"hw.optional.arm.FEAT_JSCVT");

    enable_feature(Feature::asimd, asimd);
    enable_feature(Feature::pmull, pmull);
    enable_feature(Feature::fp, fp);
    enable_feature(Feature::fp16, fp16);
    enable_feature(Feature::crc, crc);
    enable_feature(Feature::lse, lse);
    enable_feature(Feature::lse2, lse2);
    enable_feature(Feature::rdm, rdm);
    enable_feature(Feature::rcpc, rcpc);
    enable_feature(Feature::rcpc2, rcpc2);
    enable_feature(Feature::dotprod, dotprod);
    enable_feature(Feature::fhm, fhm);
    enable_feature(Feature::flagm, flagm);
    enable_feature(Feature::ssbs, ssbs);
    enable_feature(Feature::sb, sb);
    enable_feature(Feature::paca, paca);
    enable_feature(Feature::dpb, dpb);
    enable_feature(Feature::dpb2, dpb2);
    enable_feature(Feature::frintts, frintts);
    enable_feature(Feature::i8mm, i8mm);
    enable_feature(Feature::bf16, bf16);
    enable_feature(Feature::bti, bti);
    enable_feature(Feature::fcma, fcma);
    enable_feature(Feature::aes, aes && pmull);
    enable_feature(Feature::jsconv, jsconv);
    enable_feature(Feature::sha2, sha1 && sha2 && asimd);
    enable_feature(Feature::sha3, sha512 && sha3 && asimd);

    value
}
