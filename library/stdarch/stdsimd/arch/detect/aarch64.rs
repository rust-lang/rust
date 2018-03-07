//! Run-time feature detection on ARM Aarch64.

use super::bit;
use super::cache;
use super::linux;

#[macro_export]
#[unstable(feature = "stdsimd", issue = "0")]
macro_rules! is_aarch64_feature_detected {
    ("neon") => {
        // FIXME: this should be removed once we rename Aarch64 neon to asimd
        $crate::arch::detect::check_for($crate::arch::detect::Feature::asimd)
    };
    ("asimd") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::asimd)
    };
    ("pmull") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::pmull)
    };
    ("fp") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::fp)
    };
    ("fp16") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::fp16)
    };
    ("sve") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::sve)
    };
    ("crc") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::crc)
    };
    ("crypto") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::crypto)
    };
    ("lse") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::lse)
    };
    ("rdm") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::rdm)
    };
    ("rcpc") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::rcpc)
    };
    ("dotprod") => {
        $crate::arch::detect::check_for($crate::arch::detect::Feature::dotprod)
    };
    ("ras") => {
        compile_error!("\"ras\" feature cannot be detected at run-time")
    };
    ("v8.1a") => {
        compile_error!("\"v8.1a\" feature cannot be detected at run-time")
    };
    ("v8.2a") => {
        compile_error!("\"v8.2a\" feature cannot be detected at run-time")
    };
    ("v8.3a") => {
        compile_error!("\"v8.3a\" feature cannot be detected at run-time")
    };
    ($t:tt) => { compile_error!(concat!("unknown arm target feature: ", $t)) };
}

/// ARM Aarch64 CPU Feature enum. Each variant denotes a position in a bitset
/// for a particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum Feature {
    /// ARM Advanced SIMD (ASIMD)
    asimd,
    /// Polynomial Multiply
    pmull,
    /// Floating point support
    fp,
    /// Half-float support.
    fp16,
    /// Scalable Vector Extension (SVE)
    sve,
    /// CRC32 (Cyclic Redundancy Check)
    crc,
    /// Crypto: AES + PMULL + SHA1 + SHA2
    crypto,
    /// Atomics (Large System Extension)
    lse,
    /// Rounding Double Multiply (ASIMDRDM)
    rdm,
    /// Release consistent Processor consistent (RcPc)
    rcpc,
    /// Vector Dot-Product (ASIMDDP)
    dotprod,
}

pub fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    fill_features(&mut value);
    return value
}

fn fill_features(value: &mut cache::Initializer) {
    let mut enable_feature = |f, enable| {
        if enable {
            value.set(f as u32);
        }
    };

    // The values are part of the platform-specific [asm/hwcap.h][hwcap]
    //
    // [hwcap]: https://github.com/torvalds/linux/blob/master/arch/arm64/include/uapi/asm/hwcap.h
    if let Ok(auxv) = linux::auxv() {
        let fp = bit::test(auxv.hwcap, 0);
        let asimd = bit::test(auxv.hwcap, 1);
        // let evtstrm = bit::test(auxv.hwcap, 2);
        let aes = bit::test(auxv.hwcap, 3);
        let pmull = bit::test(auxv.hwcap, 4);
        let sha1 = bit::test(auxv.hwcap, 5);
        let sha2 = bit::test(auxv.hwcap, 6);
        let crc32 = bit::test(auxv.hwcap, 7);
        let atomics = bit::test(auxv.hwcap, 8);
        let fphp = bit::test(auxv.hwcap, 9);
        let asimdhp = bit::test(auxv.hwcap, 10);
        // let cpuid = bit::test(auxv.hwcap, 11);
        let asimdrdm = bit::test(auxv.hwcap, 12);
        // let jscvt = bit::test(auxv.hwcap, 13);
        // let fcma = bit::test(auxv.hwcap, 14);
        let lrcpc = bit::test(auxv.hwcap, 15);
        // let dcpop = bit::test(auxv.hwcap, 16);
        // let sha3 = bit::test(auxv.hwcap, 17);
        // let sm3 = bit::test(auxv.hwcap, 18);
        // let sm4 = bit::test(auxv.hwcap, 19);
        let asimddp = bit::test(auxv.hwcap, 20);
        // let sha512 = bit::test(auxv.hwcap, 21);
        let sve = bit::test(auxv.hwcap, 22);

        // The features are enabled approximately like in LLVM host feature detection:
        // https://github.com/llvm-mirror/llvm/blob/master/lib/Support/Host.cpp#L1273

        enable_feature(Feature::fp, fp);
        // Half-float support requires float support
        enable_feature(Feature::fp16, fp && fphp);
        enable_feature(Feature::pmull, pmull);
        enable_feature(Feature::crc, crc32);
        enable_feature(Feature::lse, atomics);
        enable_feature(Feature::rcpc, lrcpc);

        // SIMD support requires float support. If half-floats are supported,
        // SIMD support also requires half-float support
        let asimd = fp && asimd && (!fphp | asimdhp);
        enable_feature(Feature::asimd, asimd);
        // SIMD extensions require SIMD support:
        enable_feature(Feature::rdm, asimdrdm && asimd);
        enable_feature(Feature::dotprod, asimddp && asimd);
        enable_feature(Feature::sve, sve && asimd);

        // Crypto is specified as AES + PMULL + SHA1 + SHA2 per LLVM/hosts.cpp
        enable_feature(Feature::crypto, aes && pmull && sha1 && sha2);
        return
    }

    // FIXME: the logic for enabling features should be unified with auxv.
    if let Ok(c) = linux::CpuInfo::new() {
        let f = &c.field("Features");

        // 64-bit names. FIXME: In 32-bit compatibility mode /proc/cpuinfo will
        // map some of the 64-bit names to some 32-bit feature names. This does not
        // cover that yet.
        let fp = f.has("fp");
        let asimd = f.has("asimd");
        // let evtstrm = f.has("evtstrm");
        let aes = f.has("aes");
        let pmull = f.has("pmull");
        let sha1 = f.has("sha1");
        let sha2 = f.has("sha2");
        let crc32 = f.has("crc32");
        let atomics = f.has("atomics");
        let fphp = f.has("fphp");
        let asimdhp = f.has("asimdhp");
        // let cpuid = f.has("cpuid");
        let asimdrdm = f.has("asimdrdm");
        // let jscvt = f.has("jscvt");
        // let fcma = f.has("fcma");
        let lrcpc = f.has("lrcpc");
        // let dcpop = f.has("dcpop");
        // let sha3 = f.has("sha3");
        // let sm3 = f.has("sm3");
        // let sm4 = f.has("sm4");
        let asimddp = f.has("asimddp");
        // let sha512 = f.has("sha512");
        let sve = f.has("sve");

        enable_feature(Feature::fp, fp);
        enable_feature(Feature::fp16, fp && fphp);
        enable_feature(Feature::pmull, pmull);
        enable_feature(Feature::crc, crc32);
        enable_feature(Feature::lse, atomics);
        enable_feature(Feature::rcpc, lrcpc);

        let asimd = if fphp {
            fp && fphp && asimd && asimdhp
        } else {
            fp && asimd
        };
        enable_feature(Feature::asimd, asimd);
        enable_feature(Feature::rdm, asimdrdm && asimd);
        enable_feature(Feature::dotprod, asimddp && asimd);
        enable_feature(Feature::sve, sve && asimd);

        enable_feature(Feature::crypto, aes && pmull && sha1 && sha2);
        return
    }
}
