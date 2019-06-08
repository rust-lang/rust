//! Run-time feature detection for Aarch64 on Linux.

use super::{auxvec, cpuinfo};
use crate::detect::{bit, cache, Feature};

/// Performs run-time feature detection.
#[inline]
pub fn check_for(x: Feature) -> bool {
    cache::test(x as u32, detect_features)
}

/// Try to read the features from the auxiliary vector, and if that fails, try
/// to read them from /proc/cpuinfo.
fn detect_features() -> cache::Initializer {
    if let Ok(auxv) = auxvec::auxv() {
        let hwcap: AtHwcap = auxv.into();
        return hwcap.cache();
    }
    if let Ok(c) = cpuinfo::CpuInfo::new() {
        let hwcap: AtHwcap = c.into();
        return hwcap.cache();
    }
    cache::Initializer::default()
}

/// These values are part of the platform-specific [asm/hwcap.h][hwcap] .
///
/// [hwcap]: https://github.com/torvalds/linux/blob/master/arch/arm64/include/uapi/asm/hwcap.h
struct AtHwcap {
    fp: bool,    // 0
    asimd: bool, // 1
    // evtstrm: bool, // 2
    aes: bool,     // 3
    pmull: bool,   // 4
    sha1: bool,    // 5
    sha2: bool,    // 6
    crc32: bool,   // 7
    atomics: bool, // 8
    fphp: bool,    // 9
    asimdhp: bool, // 10
    // cpuid: bool, // 11
    asimdrdm: bool, // 12
    // jscvt: bool, // 13
    // fcma: bool, // 14
    lrcpc: bool, // 15
    // dcpop: bool, // 16
    // sha3: bool, // 17
    // sm3: bool, // 18
    // sm4: bool, // 19
    asimddp: bool, // 20
    // sha512: bool, // 21
    sve: bool, // 22
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
            // jscvt: bit::test(auxv.hwcap, 13),
            // fcma: bit::test(auxv.hwcap, 14),
            lrcpc: bit::test(auxv.hwcap, 15),
            // dcpop: bit::test(auxv.hwcap, 16),
            // sha3: bit::test(auxv.hwcap, 17),
            // sm3: bit::test(auxv.hwcap, 18),
            // sm4: bit::test(auxv.hwcap, 19),
            asimddp: bit::test(auxv.hwcap, 20),
            // sha512: bit::test(auxv.hwcap, 21),
            sve: bit::test(auxv.hwcap, 22),
        }
    }
}

impl From<cpuinfo::CpuInfo> for AtHwcap {
    /// Reads AtHwcap from /proc/cpuinfo .
    fn from(c: cpuinfo::CpuInfo) -> Self {
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
            // jscvt: f.has("jscvt"),
            // fcma: f.has("fcma"),
            lrcpc: f.has("lrcpc"),
            // dcpop: f.has("dcpop"),
            // sha3: f.has("sha3"),
            // sm3: f.has("sm3"),
            // sm4: f.has("sm4"),
            asimddp: f.has("asimddp"),
            // sha512: f.has("sha512"),
            sve: f.has("sve"),
        }
    }
}

impl AtHwcap {
    /// Initializes the cache from the feature -bits.
    ///
    /// The features are enabled approximately like in LLVM host feature detection:
    /// https://github.com/llvm-mirror/llvm/blob/master/lib/Support/Host.cpp#L1273
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
            enable_feature(Feature::pmull, self.pmull);
            enable_feature(Feature::crc, self.crc32);
            enable_feature(Feature::lse, self.atomics);
            enable_feature(Feature::rcpc, self.lrcpc);

            // SIMD support requires float support - if half-floats are
            // supported, it also requires half-float support:
            let asimd = self.fp && self.asimd && (!self.fphp | self.asimdhp);
            enable_feature(Feature::asimd, asimd);
            // SIMD extensions require SIMD support:
            enable_feature(Feature::rdm, self.asimdrdm && asimd);
            enable_feature(Feature::dotprod, self.asimddp && asimd);
            enable_feature(Feature::sve, self.sve && asimd);

            // Crypto is specified as AES + PMULL + SHA1 + SHA2 per LLVM/hosts.cpp
            enable_feature(
                Feature::crypto,
                self.aes && self.pmull && self.sha1 && self.sha2,
            );
        }
        value
    }
}
