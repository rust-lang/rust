//! Run-time feature detection for Aarch64 on Linux.

use super::auxvec;
use crate::detect::{bit, cache, Feature};

/// Try to read the features from the auxiliary vector, and if that fails, try
/// to read them from /proc/cpuinfo.
pub(crate) fn detect_features() -> cache::Initializer {
    #[cfg(target_os = "android")]
    let is_exynos9810 = {
        // Samsung Exynos 9810 has a bug that big and little cores have different
        // ISAs. And on older Android (pre-9), the kernel incorrectly reports
        // that features available only on some cores are available on all cores.
        // https://reviews.llvm.org/D114523
        let mut arch = [0_u8; libc::PROP_VALUE_MAX as usize];
        let len = unsafe {
            libc::__system_property_get(
                b"ro.arch\0".as_ptr() as *const libc::c_char,
                arch.as_mut_ptr() as *mut libc::c_char,
            )
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
    #[cfg(feature = "std_detect_file_io")]
    if let Ok(c) = super::cpuinfo::CpuInfo::new() {
        let hwcap: AtHwcap = c.into();
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
    // svepmull: No LLVM support.
    svebitperm: bool,
    svesha3: bool,
    svesm4: bool,
    // flagm2: No LLVM support.
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
            dcpodp: bit::test(auxv.hwcap2, 0),
            sve2: bit::test(auxv.hwcap2, 1),
            sveaes: bit::test(auxv.hwcap2, 2),
            // svepmull: bit::test(auxv.hwcap2, 3),
            svebitperm: bit::test(auxv.hwcap2, 4),
            svesha3: bit::test(auxv.hwcap2, 5),
            svesm4: bit::test(auxv.hwcap2, 6),
            // flagm2: bit::test(auxv.hwcap2, 7),
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
                enable_feature(Feature::aes, self.aes && asimd);
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
            enable_feature(Feature::rcpc, self.lrcpc);
            // RCPC2 (rcpc-immo in LLVM) requires RCPC support
            enable_feature(Feature::rcpc2, self.ilrcpc && self.lrcpc);
            enable_feature(Feature::dit, self.dit);
            enable_feature(Feature::flagm, self.flagm);
            enable_feature(Feature::ssbs, self.ssbs);
            enable_feature(Feature::sb, self.sb);
            enable_feature(Feature::paca, self.paca);
            enable_feature(Feature::pacg, self.pacg);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std_detect_file_io")]
    mod auxv_from_file {
        use super::auxvec::auxv_from_file;
        use super::*;
        // The baseline hwcaps used in the (artificial) auxv test files.
        fn baseline_hwcaps() -> AtHwcap {
            AtHwcap {
                fp: true,
                asimd: true,
                aes: true,
                pmull: true,
                sha1: true,
                sha2: true,
                crc32: true,
                atomics: true,
                fphp: true,
                asimdhp: true,
                asimdrdm: true,
                lrcpc: true,
                dcpop: true,
                asimddp: true,
                ssbs: true,
                ..AtHwcap::default()
            }
        }

        #[test]
        fn linux_empty_hwcap2_aarch64() {
            let file = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/detect/test_data/linux-empty-hwcap2-aarch64.auxv"
            );
            println!("file: {file}");
            let v = auxv_from_file(file).unwrap();
            println!("HWCAP : 0x{:0x}", v.hwcap);
            println!("HWCAP2: 0x{:0x}", v.hwcap2);
            assert_eq!(AtHwcap::from(v), baseline_hwcaps());
        }
        #[test]
        fn linux_no_hwcap2_aarch64() {
            let file = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/detect/test_data/linux-no-hwcap2-aarch64.auxv"
            );
            println!("file: {file}");
            let v = auxv_from_file(file).unwrap();
            println!("HWCAP : 0x{:0x}", v.hwcap);
            println!("HWCAP2: 0x{:0x}", v.hwcap2);
            assert_eq!(AtHwcap::from(v), baseline_hwcaps());
        }
        #[test]
        fn linux_hwcap2_aarch64() {
            let file = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/detect/test_data/linux-hwcap2-aarch64.auxv"
            );
            println!("file: {file}");
            let v = auxv_from_file(file).unwrap();
            println!("HWCAP : 0x{:0x}", v.hwcap);
            println!("HWCAP2: 0x{:0x}", v.hwcap2);
            assert_eq!(
                AtHwcap::from(v),
                AtHwcap {
                    // Some other HWCAP bits.
                    paca: true,
                    pacg: true,
                    // HWCAP2-only bits.
                    dcpodp: true,
                    frint: true,
                    rng: true,
                    bti: true,
                    mte: true,
                    ..baseline_hwcaps()
                }
            );
        }
    }
}
