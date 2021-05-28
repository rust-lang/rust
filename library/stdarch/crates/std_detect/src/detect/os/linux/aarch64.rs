//! Run-time feature detection for Aarch64 on Linux.

use super::auxvec;
use crate::detect::{bit, cache, Feature};

/// Try to read the features from the auxiliary vector, and if that fails, try
/// to read them from /proc/cpuinfo.
pub(crate) fn detect_features() -> cache::Initializer {
    if let Ok(auxv) = auxvec::auxv() {
        let hwcap: AtHwcap = auxv.into();
        return hwcap.cache();
    }
    #[cfg(feature = "std_detect_file_io")]
    if let Ok(c) = super::cpuinfo::CpuInfo::new() {
        let hwcap: AtHwcap = c.into();
        return hwcap.cache();
    }
    cache::Initializer::default()
}

/// These values are part of the platform-specific [asm/hwcap.h][hwcap] .
///
/// The names match those used for cpuinfo.
///
/// [hwcap]: https://github.com/torvalds/linux/blob/master/arch/arm64/include/uapi/asm/hwcap.h
struct AtHwcap {
    fp: bool,    // 0
    asimd: bool, // 1
    // evtstrm: bool, // 2 No LLVM support
    aes: bool,     // 3
    pmull: bool,   // 4
    sha1: bool,    // 5
    sha2: bool,    // 6
    crc32: bool,   // 7
    atomics: bool, // 8
    fphp: bool,    // 9
    asimdhp: bool, // 10
    // cpuid: bool, // 11 No LLVM support
    asimdrdm: bool, // 12
    jscvt: bool,    // 13
    fcma: bool,     // 14
    lrcpc: bool,    // 15
    dcpop: bool,    // 16
    sha3: bool,     // 17
    sm3: bool,      // 18
    sm4: bool,      // 19
    asimddp: bool,  // 20
    sha512: bool,   // 21
    sve: bool,      // 22
    fhm: bool,      // 23
    dit: bool,      // 24
    uscat: bool,    // 25
    ilrcpc: bool,   // 26
    flagm: bool,    // 27
    ssbs: bool,     // 28
    sb: bool,       // 29
    paca: bool,     // 30
    pacg: bool,     // 31
    dcpodp: bool,   // 32
    sve2: bool,     // 33
    sveaes: bool,   // 34
    // svepmull: bool, // 35 No LLVM support
    svebitperm: bool, // 36
    svesha3: bool,    // 37
    svesm4: bool,     // 38
    // flagm2: bool, // 39 No LLVM support
    frint: bool, // 40
    // svei8mm: bool, // 41 See i8mm feature
    svef32mm: bool, // 42
    svef64mm: bool, // 43
    // svebf16: bool, // 44 See bf16 feature
    i8mm: bool, // 45
    bf16: bool, // 46
    // dgh: bool, // 47 No LLVM support
    rng: bool, // 48
    bti: bool, // 49
    mte: bool, // 50
}

impl From<auxvec::AuxVec> for AtHwcap {
    /// Reads AtHwcap from the auxiliary vector.
    fn from(auxv: auxvec::AuxVec) -> Self {
        AtHwcap {
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
            dcpodp: bit::test(auxv.hwcap, 32),
            sve2: bit::test(auxv.hwcap, 33),
            sveaes: bit::test(auxv.hwcap, 34),
            // svepmull: bit::test(auxv.hwcap, 35),
            svebitperm: bit::test(auxv.hwcap, 36),
            svesha3: bit::test(auxv.hwcap, 37),
            svesm4: bit::test(auxv.hwcap, 38),
            // flagm2: bit::test(auxv.hwcap, 39),
            frint: bit::test(auxv.hwcap, 40),
            // svei8mm: bit::test(auxv.hwcap, 41),
            svef32mm: bit::test(auxv.hwcap, 42),
            svef64mm: bit::test(auxv.hwcap, 43),
            // svebf16: bit::test(auxv.hwcap, 44),
            i8mm: bit::test(auxv.hwcap, 45),
            bf16: bit::test(auxv.hwcap, 46),
            // dgh: bit::test(auxv.hwcap, 47),
            rng: bit::test(auxv.hwcap, 48),
            bti: bit::test(auxv.hwcap, 49),
            mte: bit::test(auxv.hwcap, 50),
        }
    }
}

#[cfg(feature = "std_detect_file_io")]
impl From<super::cpuinfo::CpuInfo> for AtHwcap {
    /// Reads AtHwcap from /proc/cpuinfo .
    fn from(c: super::cpuinfo::CpuInfo) -> Self {
        let f = &c.field("Features");
        AtHwcap {
            // 64-bit names. FIXME: In 32-bit compatibility mode /proc/cpuinfo will
            // map some of the 64-bit names to some 32-bit feature names. This does not
            // cover that yet.
            fp: f.has("fp"),
            asimd: f.has("asimd"),
            // evtstrm: f.has("evtstrm"),
            aes: f.has("aes"),
            pmull: f.has("pmull"),
            sha1: f.has("sha1"),
            sha2: f.has("sha2"),
            crc32: f.has("crc32"),
            atomics: f.has("atomics"),
            fphp: f.has("fphp"),
            asimdhp: f.has("asimdhp"),
            // cpuid: f.has("cpuid"),
            asimdrdm: f.has("asimdrdm"),
            jscvt: f.has("jscvt"),
            fcma: f.has("fcma"),
            lrcpc: f.has("lrcpc"),
            dcpop: f.has("dcpop"),
            sha3: f.has("sha3"),
            sm3: f.has("sm3"),
            sm4: f.has("sm4"),
            asimddp: f.has("asimddp"),
            sha512: f.has("sha512"),
            sve: f.has("sve"),
            fhm: f.has("asimdfhm"),
            dit: f.has("dit"),
            uscat: f.has("uscat"),
            ilrcpc: f.has("ilrcpc"),
            flagm: f.has("flagm"),
            ssbs: f.has("ssbs"),
            sb: f.has("sb"),
            paca: f.has("paca"),
            pacg: f.has("pacg"),
            dcpodp: f.has("dcpodp"),
            sve2: f.has("sve2"),
            sveaes: f.has("sveaes"),
            // svepmull: f.has("svepmull"),
            svebitperm: f.has("svebitperm"),
            svesha3: f.has("svesha3"),
            svesm4: f.has("svesm4"),
            // flagm2: f.has("flagm2"),
            frint: f.has("frint"),
            // svei8mm: f.has("svei8mm"),
            svef32mm: f.has("svef32mm"),
            svef64mm: f.has("svef64mm"),
            // svebf16: f.has("svebf16"),
            i8mm: f.has("i8mm"),
            bf16: f.has("bf16"),
            // dgh: f.has("dgh"),
            rng: f.has("rng"),
            bti: f.has("bti"),
            mte: f.has("mte"),
        }
    }
}

impl AtHwcap {
    /// Initializes the cache from the feature -bits.
    ///
    /// The feature dependencies here come directly from LLVM's feature definintions:
    /// https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AArch64/AArch64.td
    fn cache(self) -> cache::Initializer {
        let mut value = cache::Initializer::default();
        {
            let mut enable_feature = |f, enable| {
                if enable {
                    value.set(f as u32);
                }
            };

            enable_feature(Feature::fp, self.fp);
            // Half-float support requires float support
            enable_feature(Feature::fp16, self.fp && self.fphp);
            // FHM (fp16fml in LLVM) requires half float support
            enable_feature(Feature::fhm, self.fphp && self.fhm);
            enable_feature(Feature::pmull, self.pmull);
            enable_feature(Feature::crc, self.crc32);
            enable_feature(Feature::lse, self.atomics);
            enable_feature(Feature::lse2, self.uscat);
            enable_feature(Feature::rcpc, self.lrcpc);
            // RCPC2 (rcpc-immo in LLVM) requires RCPC support
            enable_feature(Feature::rcpc2, self.ilrcpc && self.lrcpc);
            enable_feature(Feature::dit, self.dit);
            enable_feature(Feature::flagm, self.flagm);
            enable_feature(Feature::ssbs, self.ssbs);
            enable_feature(Feature::sb, self.sb);
            // FEAT_PAuth provides both paca & pacg
            enable_feature(Feature::pauth, self.paca && self.pacg);
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
            // SVE2 extensions require SVE2 and crypto features
            enable_feature(Feature::sve2_aes, self.sveaes && sve2 && self.aes);
            enable_feature(
                Feature::sve2_sm4,
                self.svesm4 && sve2 && self.sm3 && self.sm4,
            );
            enable_feature(
                Feature::sve2_sha3,
                self.svesha3 && sve2 && self.sha512 && self.sha3 && self.sha1 && self.sha2,
            );
            enable_feature(Feature::sve2_bitperm, self.svebitperm && self.sve2);
        }
        value
    }
}
