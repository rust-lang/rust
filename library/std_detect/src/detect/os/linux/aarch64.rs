//! Run-time feature detection for Aarch64 on Linux.

use super::auxvec;
use crate::detect::{Feature, bit, cache};

/// Try to read the features from the auxiliary vector.
pub(crate) fn detect_features() -> cache::Initializer {
    #[cfg(target_os = "android")]
    let is_exynos9810 = {
        // Samsung Exynos 9810 has a bug that big and little cores have different
        // ISAs. And on older Android (pre-9), the kernel incorrectly reports
        // that features available only on some cores are available on all cores.
        // https://reviews.llvm.org/D114523
        let mut arch = [0_u8; libc::PROP_VALUE_MAX as usize];
        let len = unsafe {
            libc::__system_property_get(c"ro.arch".as_ptr(), arch.as_mut_ptr() as *mut libc::c_char)
        };
        // On Exynos, ro.arch is not available on Android 12+, but it is fine
        // because Android 9+ includes the fix.
        len > 0 && arch.starts_with(b"exynos9810")
    };
    #[cfg(not(target_os = "android"))]
    let is_exynos9810 = false;

    if let Ok(auxv) = auxvec::auxv() {
        let hwcap: AtHwcap = auxv.into();
        return hwcap.cache(is_exynos9810);
    }
    cache::Initializer::default()
}

/// These values are part of the platform-specific [asm/hwcap.h][hwcap] .
///
/// The names match those used for cpuinfo.
///
/// [hwcap]: https://github.com/torvalds/linux/blob/master/arch/arm64/include/uapi/asm/hwcap.h
#[derive(Debug, Default, PartialEq)]
struct AtHwcap {
    // AT_HWCAP
    fp: bool,
    asimd: bool,
    // evtstrm: No LLVM support.
    aes: bool,
    pmull: bool,
    sha1: bool,
    sha2: bool,
    crc32: bool,
    atomics: bool,
    fphp: bool,
    asimdhp: bool,
    // cpuid: No LLVM support.
    asimdrdm: bool,
    jscvt: bool,
    fcma: bool,
    lrcpc: bool,
    dcpop: bool,
    sha3: bool,
    sm3: bool,
    sm4: bool,
    asimddp: bool,
    sha512: bool,
    sve: bool,
    fhm: bool,
    dit: bool,
    uscat: bool,
    ilrcpc: bool,
    flagm: bool,
    ssbs: bool,
    sb: bool,
    paca: bool,
    pacg: bool,

    // AT_HWCAP2
    dcpodp: bool,
    sve2: bool,
    sveaes: bool,
    svepmull: bool,
    svebitperm: bool,
    svesha3: bool,
    svesm4: bool,
    flagm2: bool,
    frint: bool,
    // svei8mm: See i8mm feature.
    svef32mm: bool,
    svef64mm: bool,
    // svebf16: See bf16 feature.
    i8mm: bool,
    bf16: bool,
    // dgh: No LLVM support.
    rng: bool,
    bti: bool,
    mte: bool,
    ecv: bool,
    // afp: bool,
    // rpres: bool,
    // mte3: bool,
    sme: bool,
    smei16i64: bool,
    smef64f64: bool,
    // smei8i32: bool,
    // smef16f32: bool,
    // smeb16f32: bool,
    // smef32f32: bool,
    smefa64: bool,
    wfxt: bool,
    // ebf16: bool,
    // sveebf16: bool,
    cssc: bool,
    // rprfm: bool,
    sve2p1: bool,
    sme2: bool,
    sme2p1: bool,
    // smei16i32: bool,
    // smebi32i32: bool,
    smeb16b16: bool,
    smef16f16: bool,
    mops: bool,
    hbc: bool,
    sveb16b16: bool,
    lrcpc3: bool,
    lse128: bool,
    fpmr: bool,
    lut: bool,
    faminmax: bool,
    f8cvt: bool,
    f8fma: bool,
    f8dp4: bool,
    f8dp2: bool,
    f8e4m3: bool,
    f8e5m2: bool,
    smelutv2: bool,
    smef8f16: bool,
    smef8f32: bool,
    smesf8fma: bool,
    smesf8dp4: bool,
    smesf8dp2: bool,
    // pauthlr: bool,
}

impl From<auxvec::AuxVec> for AtHwcap {
    /// Reads AtHwcap from the auxiliary vector.
    fn from(auxv: auxvec::AuxVec) -> Self {
        let mut cap = AtHwcap {
            fp: bit::test(auxv.hwcap, 0),
            asimd: bit::test(auxv.hwcap, 1),
            // evtstrm: bit::test(auxv.hwcap, 2),
            aes: bit::test(auxv.hwcap, 3),
            pmull: bit::test(auxv.hwcap, 4),
            sha1: bit::test(auxv.hwcap, 5),
            sha2: bit::test(auxv.hwcap, 6),
            crc32: bit::test(auxv.hwcap, 7),
            atomics: bit::test(auxv.hwcap, 8),
            fphp: bit::test(auxv.hwcap, 9),
            asimdhp: bit::test(auxv.hwcap, 10),
            // cpuid: bit::test(auxv.hwcap, 11),
            asimdrdm: bit::test(auxv.hwcap, 12),
            jscvt: bit::test(auxv.hwcap, 13),
            fcma: bit::test(auxv.hwcap, 14),
            lrcpc: bit::test(auxv.hwcap, 15),
            dcpop: bit::test(auxv.hwcap, 16),
            sha3: bit::test(auxv.hwcap, 17),
            sm3: bit::test(auxv.hwcap, 18),
            sm4: bit::test(auxv.hwcap, 19),
            asimddp: bit::test(auxv.hwcap, 20),
            sha512: bit::test(auxv.hwcap, 21),
            sve: bit::test(auxv.hwcap, 22),
            fhm: bit::test(auxv.hwcap, 23),
            dit: bit::test(auxv.hwcap, 24),
            uscat: bit::test(auxv.hwcap, 25),
            ilrcpc: bit::test(auxv.hwcap, 26),
            flagm: bit::test(auxv.hwcap, 27),
            ssbs: bit::test(auxv.hwcap, 28),
            sb: bit::test(auxv.hwcap, 29),
            paca: bit::test(auxv.hwcap, 30),
            pacg: bit::test(auxv.hwcap, 31),

            // AT_HWCAP2
            dcpodp: bit::test(auxv.hwcap2, 0),
            sve2: bit::test(auxv.hwcap2, 1),
            sveaes: bit::test(auxv.hwcap2, 2),
            svepmull: bit::test(auxv.hwcap2, 3),
            svebitperm: bit::test(auxv.hwcap2, 4),
            svesha3: bit::test(auxv.hwcap2, 5),
            svesm4: bit::test(auxv.hwcap2, 6),
            flagm2: bit::test(auxv.hwcap2, 7),
            frint: bit::test(auxv.hwcap2, 8),
            // svei8mm: bit::test(auxv.hwcap2, 9),
            svef32mm: bit::test(auxv.hwcap2, 10),
            svef64mm: bit::test(auxv.hwcap2, 11),
            // svebf16: bit::test(auxv.hwcap2, 12),
            i8mm: bit::test(auxv.hwcap2, 13),
            bf16: bit::test(auxv.hwcap2, 14),
            // dgh: bit::test(auxv.hwcap2, 15),
            rng: bit::test(auxv.hwcap2, 16),
            bti: bit::test(auxv.hwcap2, 17),
            mte: bit::test(auxv.hwcap2, 18),
            ecv: bit::test(auxv.hwcap2, 19),
            // afp: bit::test(auxv.hwcap2, 20),
            // rpres: bit::test(auxv.hwcap2, 21),
            // mte3: bit::test(auxv.hwcap2, 22),
            sme: bit::test(auxv.hwcap2, 23),
            smei16i64: bit::test(auxv.hwcap2, 24),
            smef64f64: bit::test(auxv.hwcap2, 25),
            // smei8i32: bit::test(auxv.hwcap2, 26),
            // smef16f32: bit::test(auxv.hwcap2, 27),
            // smeb16f32: bit::test(auxv.hwcap2, 28),
            // smef32f32: bit::test(auxv.hwcap2, 29),
            smefa64: bit::test(auxv.hwcap2, 30),
            wfxt: bit::test(auxv.hwcap2, 31),
            ..Default::default()
        };

        // Hardware capabilities from bits 32 to 63 should only
        // be tested on LP64 targets with 64 bits `usize`.
        // On ILP32 targets like `aarch64-unknown-linux-gnu_ilp32`,
        // these hardware capabilities will default to `false`.
        // https://github.com/rust-lang/rust/issues/146230
        #[cfg(target_pointer_width = "64")]
        {
            // cap.ebf16: bit::test(auxv.hwcap2, 32);
            // cap.sveebf16: bit::test(auxv.hwcap2, 33);
            cap.cssc = bit::test(auxv.hwcap2, 34);
            // cap.rprfm: bit::test(auxv.hwcap2, 35);
            cap.sve2p1 = bit::test(auxv.hwcap2, 36);
            cap.sme2 = bit::test(auxv.hwcap2, 37);
            cap.sme2p1 = bit::test(auxv.hwcap2, 38);
            // cap.smei16i32 = bit::test(auxv.hwcap2, 39);
            // cap.smebi32i32 = bit::test(auxv.hwcap2, 40);
            cap.smeb16b16 = bit::test(auxv.hwcap2, 41);
            cap.smef16f16 = bit::test(auxv.hwcap2, 42);
            cap.mops = bit::test(auxv.hwcap2, 43);
            cap.hbc = bit::test(auxv.hwcap2, 44);
            cap.sveb16b16 = bit::test(auxv.hwcap2, 45);
            cap.lrcpc3 = bit::test(auxv.hwcap2, 46);
            cap.lse128 = bit::test(auxv.hwcap2, 47);
            cap.fpmr = bit::test(auxv.hwcap2, 48);
            cap.lut = bit::test(auxv.hwcap2, 49);
            cap.faminmax = bit::test(auxv.hwcap2, 50);
            cap.f8cvt = bit::test(auxv.hwcap2, 51);
            cap.f8fma = bit::test(auxv.hwcap2, 52);
            cap.f8dp4 = bit::test(auxv.hwcap2, 53);
            cap.f8dp2 = bit::test(auxv.hwcap2, 54);
            cap.f8e4m3 = bit::test(auxv.hwcap2, 55);
            cap.f8e5m2 = bit::test(auxv.hwcap2, 56);
            cap.smelutv2 = bit::test(auxv.hwcap2, 57);
            cap.smef8f16 = bit::test(auxv.hwcap2, 58);
            cap.smef8f32 = bit::test(auxv.hwcap2, 59);
            cap.smesf8fma = bit::test(auxv.hwcap2, 60);
            cap.smesf8dp4 = bit::test(auxv.hwcap2, 61);
            cap.smesf8dp2 = bit::test(auxv.hwcap2, 62);
            // cap.pauthlr = bit::test(auxv.hwcap2, ??);
        }
        cap
    }
}

impl AtHwcap {
    /// Initializes the cache from the feature -bits.
    ///
    /// The feature dependencies here come directly from LLVM's feature definitions:
    /// https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AArch64/AArch64.td
    fn cache(self, is_exynos9810: bool) -> cache::Initializer {
        let mut value = cache::Initializer::default();
        {
            let mut enable_feature = |f, enable| {
                if enable {
                    value.set(f as u32);
                }
            };

            // Samsung Exynos 9810 has a bug that big and little cores have different
            // ISAs. And on older Android (pre-9), the kernel incorrectly reports
            // that features available only on some cores are available on all cores.
            // So, only check features that are known to be available on exynos-m3:
            // $ rustc --print cfg --target aarch64-linux-android -C target-cpu=exynos-m3 | grep target_feature
            // See also https://github.com/rust-lang/stdarch/pull/1378#discussion_r1103748342.
            if is_exynos9810 {
                enable_feature(Feature::fp, self.fp);
                enable_feature(Feature::crc, self.crc32);
                // ASIMD support requires float support - if half-floats are
                // supported, it also requires half-float support:
                let asimd = self.fp && self.asimd && (!self.fphp | self.asimdhp);
                enable_feature(Feature::asimd, asimd);
                // Cryptographic extensions require ASIMD
                // AES also covers FEAT_PMULL
                enable_feature(Feature::aes, self.aes && self.pmull && asimd);
                enable_feature(Feature::sha2, self.sha1 && self.sha2 && asimd);
                return value;
            }

            enable_feature(Feature::fp, self.fp);
            // Half-float support requires float support
            enable_feature(Feature::fp16, self.fp && self.fphp);
            // FHM (fp16fml in LLVM) requires half float support
            enable_feature(Feature::fhm, self.fphp && self.fhm);
            enable_feature(Feature::pmull, self.pmull);
            enable_feature(Feature::crc, self.crc32);
            enable_feature(Feature::lse, self.atomics);
            enable_feature(Feature::lse2, self.uscat);
            enable_feature(Feature::lse128, self.lse128 && self.atomics);
            enable_feature(Feature::rcpc, self.lrcpc);
            // RCPC2 (rcpc-immo in LLVM) requires RCPC support
            let rcpc2 = self.ilrcpc && self.lrcpc;
            enable_feature(Feature::rcpc2, rcpc2);
            enable_feature(Feature::rcpc3, self.lrcpc3 && rcpc2);
            enable_feature(Feature::dit, self.dit);
            enable_feature(Feature::flagm, self.flagm);
            enable_feature(Feature::flagm2, self.flagm2);
            enable_feature(Feature::ssbs, self.ssbs);
            enable_feature(Feature::sb, self.sb);
            enable_feature(Feature::paca, self.paca);
            enable_feature(Feature::pacg, self.pacg);
            // enable_feature(Feature::pauth_lr, self.pauthlr);
            enable_feature(Feature::dpb, self.dcpop);
            enable_feature(Feature::dpb2, self.dcpodp);
            enable_feature(Feature::rand, self.rng);
            enable_feature(Feature::bti, self.bti);
            enable_feature(Feature::mte, self.mte);
            // jsconv requires float support
            enable_feature(Feature::jsconv, self.jscvt && self.fp);
            enable_feature(Feature::rdm, self.asimdrdm);
            enable_feature(Feature::dotprod, self.asimddp);
            enable_feature(Feature::frintts, self.frint);

            // FEAT_I8MM & FEAT_BF16 also include optional SVE components which linux exposes
            // separately. We ignore that distinction here.
            enable_feature(Feature::i8mm, self.i8mm);
            enable_feature(Feature::bf16, self.bf16);

            // ASIMD support requires float support - if half-floats are
            // supported, it also requires half-float support:
            let asimd = self.fp && self.asimd && (!self.fphp | self.asimdhp);
            enable_feature(Feature::asimd, asimd);
            // ASIMD extensions require ASIMD support:
            enable_feature(Feature::fcma, self.fcma && asimd);
            enable_feature(Feature::sve, self.sve && asimd);

            // SVE extensions require SVE & ASIMD
            enable_feature(Feature::f32mm, self.svef32mm && self.sve && asimd);
            enable_feature(Feature::f64mm, self.svef64mm && self.sve && asimd);

            // Cryptographic extensions require ASIMD
            enable_feature(Feature::aes, self.aes && asimd);
            enable_feature(Feature::sha2, self.sha1 && self.sha2 && asimd);
            // SHA512/SHA3 require SHA1 & SHA256
            enable_feature(
                Feature::sha3,
                self.sha512 && self.sha3 && self.sha1 && self.sha2 && asimd,
            );
            enable_feature(Feature::sm4, self.sm3 && self.sm4 && asimd);

            // SVE2 requires SVE
            let sve2 = self.sve2 && self.sve && asimd;
            enable_feature(Feature::sve2, sve2);
            enable_feature(Feature::sve2p1, self.sve2p1 && sve2);
            // SVE2 extensions require SVE2 and crypto features
            enable_feature(Feature::sve2_aes, self.sveaes && self.svepmull && sve2 && self.aes);
            enable_feature(Feature::sve2_sm4, self.svesm4 && sve2 && self.sm3 && self.sm4);
            enable_feature(
                Feature::sve2_sha3,
                self.svesha3 && sve2 && self.sha512 && self.sha3 && self.sha1 && self.sha2,
            );
            enable_feature(Feature::sve2_bitperm, self.svebitperm && self.sve2);
            enable_feature(Feature::sve_b16b16, self.bf16 && self.sveb16b16);
            enable_feature(Feature::hbc, self.hbc);
            enable_feature(Feature::mops, self.mops);
            enable_feature(Feature::ecv, self.ecv);
            enable_feature(Feature::lut, self.lut);
            enable_feature(Feature::cssc, self.cssc);
            enable_feature(Feature::fpmr, self.fpmr);
            enable_feature(Feature::faminmax, self.faminmax);
            let fp8 = self.f8cvt && self.faminmax && self.lut && self.bf16;
            enable_feature(Feature::fp8, fp8);
            let fp8fma = self.f8fma && fp8;
            enable_feature(Feature::fp8fma, fp8fma);
            let fp8dot4 = self.f8dp4 && fp8fma;
            enable_feature(Feature::fp8dot4, fp8dot4);
            enable_feature(Feature::fp8dot2, self.f8dp2 && fp8dot4);
            enable_feature(Feature::wfxt, self.wfxt);
            let sme = self.sme && self.bf16;
            enable_feature(Feature::sme, sme);
            enable_feature(Feature::sme_i16i64, self.smei16i64 && sme);
            enable_feature(Feature::sme_f64f64, self.smef64f64 && sme);
            enable_feature(Feature::sme_fa64, self.smefa64 && sme && sve2);
            let sme2 = self.sme2 && sme;
            enable_feature(Feature::sme2, sme2);
            enable_feature(Feature::sme2p1, self.sme2p1 && sme2);
            enable_feature(
                Feature::sme_b16b16,
                sme2 && self.bf16 && self.sveb16b16 && self.smeb16b16,
            );
            enable_feature(Feature::sme_f16f16, self.smef16f16 && sme2);
            enable_feature(Feature::sme_lutv2, self.smelutv2);
            let sme_f8f32 = self.smef8f32 && sme2 && fp8;
            enable_feature(Feature::sme_f8f32, sme_f8f32);
            enable_feature(Feature::sme_f8f16, self.smef8f16 && sme_f8f32);
            let ssve_fp8fma = self.smesf8fma && sme2 && fp8;
            enable_feature(Feature::ssve_fp8fma, ssve_fp8fma);
            let ssve_fp8dot4 = self.smesf8dp4 && ssve_fp8fma;
            enable_feature(Feature::ssve_fp8dot4, ssve_fp8dot4);
            enable_feature(Feature::ssve_fp8dot2, self.smesf8dp2 && ssve_fp8dot4);
        }
        value
    }
}

#[cfg(target_endian = "little")]
#[cfg(test)]
mod tests;
