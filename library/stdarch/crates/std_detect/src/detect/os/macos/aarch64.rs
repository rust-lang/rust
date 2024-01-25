//! Run-time feature detection for aarch64 on macOS.

use crate::detect::{cache, Feature};

#[inline]
fn _sysctlbyname(name: &str) -> bool {
    use libc;

    let mut enabled: i32 = 0;
    let mut enabled_len: usize = 4;
    let enabled_ptr = &mut enabled as *mut i32 as *mut libc::c_void;

    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const i8,
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

    let asimd = _sysctlbyname("hw.optional.AdvSIMD\0");
    let pmull = _sysctlbyname("hw.optional.arm.FEAT_PMULL\0");
    let fp = _sysctlbyname("hw.optional.floatingpoint\0");
    let fp16 = _sysctlbyname("hw.optional.arm.FEAT_FP16\0");
    let crc = _sysctlbyname("hw.optional.armv8_crc32\0");
    let lse = _sysctlbyname("hw.optional.arm.FEAT_LSE\0");
    let lse2 = _sysctlbyname("hw.optional.arm.FEAT_LSE2\0");
    let rdm = _sysctlbyname("hw.optional.arm.FEAT_RDM\0");
    let rcpc = _sysctlbyname("hw.optional.arm.FEAT_LRCPC\0");
    let rcpc2 = _sysctlbyname("hw.optional.arm.FEAT_LRCPC2\0");
    let dotprod = _sysctlbyname("hw.optional.arm.FEAT_DotProd\0");
    let fhm = _sysctlbyname("hw.optional.arm.FEAT_FHM\0");
    let flagm = _sysctlbyname("hw.optional.arm.FEAT_FlagM\0");
    let ssbs = _sysctlbyname("hw.optional.arm.FEAT_SSBS\0");
    let sb = _sysctlbyname("hw.optional.arm.FEAT_SB\0");
    let paca = _sysctlbyname("hw.optional.arm.FEAT_PAuth\0");
    let dpb = _sysctlbyname("hw.optional.arm.FEAT_DPB\0");
    let dpb2 = _sysctlbyname("hw.optional.arm.FEAT_DPB2\0");
    let frintts = _sysctlbyname("hw.optional.arm.FEAT_FRINTTS\0");
    let i8mm = _sysctlbyname("hw.optional.arm.FEAT_I8MM\0");
    let bf16 = _sysctlbyname("hw.optional.arm.FEAT_BF16\0");
    let bti = _sysctlbyname("hw.optional.arm.FEAT_BTI\0");
    let fcma = _sysctlbyname("hw.optional.arm.FEAT_FCMA\0");
    let aes = _sysctlbyname("hw.optional.arm.FEAT_AES\0");
    let sha1 = _sysctlbyname("hw.optional.arm.FEAT_SHA1\0");
    let sha2 = _sysctlbyname("hw.optional.arm.FEAT_SHA256\0");
    let sha3 = _sysctlbyname("hw.optional.arm.FEAT_SHA3\0");
    let sha512 = _sysctlbyname("hw.optional.arm.FEAT_SHA512\0");
    let jsconv = _sysctlbyname("hw.optional.arm.FEAT_JSCVT\0");

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
