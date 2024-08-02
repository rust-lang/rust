//! x86 run-time feature detection is OS independent.

#[cfg(target_arch = "x86")]
use crate::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::*;

use crate::mem;

use crate::detect::{Feature, cache, bit};

/// Performs run-time feature detection.
#[inline]
pub fn check_for(x: Feature) -> bool {
    cache::test(x as u32, detect_features)
}

/// Run-time feature detection on x86 works by using the CPUID instruction.
///
/// The [CPUID Wikipedia page][wiki_cpuid] contains
/// all the information about which flags to set to query which values, and in
/// which registers these are reported.
///
/// The definitive references are:
/// - [Intel 64 and IA-32 Architectures Software Developer's Manual Volume 2:
///   Instruction Set Reference, A-Z][intel64_ref].
/// - [AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and
///   System Instructions][amd64_ref].
///
/// [wiki_cpuid]: https://en.wikipedia.org/wiki/CPUID
/// [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
/// [amd64_ref]: http://support.amd.com/TechDocs/24594.pdf
#[allow(clippy::similar_names)]
fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();

    // 0. EAX = 0: Basic Information:
    // - EAX returns the "Highest Function Parameter", that is, the maximum
    // leaf value for subsequent calls of `cpuinfo` in range [0,
    // 0x8000_0000]. - The vendor ID is stored in 12 u8 ascii chars,
    // returned in EBX, EDX, and   ECX (in that order):
    let (max_basic_leaf, vendor_id) = unsafe {
        let CpuidResult {
            eax: max_basic_leaf,
            ebx,
            ecx,
            edx,
        } = __cpuid(0);
        let vendor_id: [[u8; 4]; 3] = [
            mem::transmute(ebx),
            mem::transmute(edx),
            mem::transmute(ecx),
        ];
        let vendor_id: [u8; 12] = mem::transmute(vendor_id);
        (max_basic_leaf, vendor_id)
    };

    if max_basic_leaf < 1 {
        // Earlier Intel 486, CPUID not implemented
        return value;
    }

    // EAX = 1, ECX = 0: Queries "Processor Info and Feature Bits";
    // Contains information about most x86 features.
    let CpuidResult {
        ecx: proc_info_ecx,
        edx: proc_info_edx,
        ..
    } = unsafe { __cpuid(0x0000_0001_u32) };

    // EAX = 7, ECX = 0: Queries "Extended Features";
    // Contains information about bmi,bmi2, and avx2 support.
    let (extended_features_ebx, extended_features_ecx) = if max_basic_leaf >= 7
    {
        let CpuidResult { ebx, ecx, .. } = unsafe { __cpuid(0x0000_0007_u32) };
        (ebx, ecx)
    } else {
        (0, 0) // CPUID does not support "Extended Features"
    };

    // EAX = 0x8000_0000, ECX = 0: Get Highest Extended Function Supported
    // - EAX returns the max leaf value for extended information, that is,
    // `cpuid` calls in range [0x8000_0000; u32::MAX]:
    let CpuidResult {
        eax: extended_max_basic_leaf,
        ..
    } = unsafe { __cpuid(0x8000_0000_u32) };

    // EAX = 0x8000_0001, ECX=0: Queries "Extended Processor Info and Feature
    // Bits"
    let extended_proc_info_ecx = if extended_max_basic_leaf >= 1 {
        let CpuidResult { ecx, .. } = unsafe { __cpuid(0x8000_0001_u32) };
        ecx
    } else {
        0
    };

    {
        // borrows value till the end of this scope:
        let mut enable = |r, rb, f| {
            if bit::test(r as usize, rb) {
                value.set(f as u32);
            }
        };

        enable(proc_info_ecx, 0, Feature::sse3);
        enable(proc_info_ecx, 1, Feature::pclmulqdq);
        enable(proc_info_ecx, 9, Feature::ssse3);
        enable(proc_info_ecx, 13, Feature::cmpxchg16b);
        enable(proc_info_ecx, 19, Feature::sse4_1);
        enable(proc_info_ecx, 20, Feature::sse4_2);
        enable(proc_info_ecx, 23, Feature::popcnt);
        enable(proc_info_ecx, 25, Feature::aes);
        enable(proc_info_ecx, 29, Feature::f16c);
        enable(proc_info_ecx, 30, Feature::rdrand);
        enable(extended_features_ebx, 18, Feature::rdseed);
        enable(extended_features_ebx, 19, Feature::adx);
        enable(extended_features_ebx, 11, Feature::rtm);
        enable(proc_info_edx, 4, Feature::tsc);
        enable(proc_info_edx, 23, Feature::mmx);
        enable(proc_info_edx, 24, Feature::fxsr);
        enable(proc_info_edx, 25, Feature::sse);
        enable(proc_info_edx, 26, Feature::sse2);
        enable(extended_features_ebx, 29, Feature::sha);

        enable(extended_features_ebx, 3, Feature::bmi);
        enable(extended_features_ebx, 8, Feature::bmi2);

        // `XSAVE` and `AVX` support:
        let cpu_xsave = bit::test(proc_info_ecx as usize, 26);
        if cpu_xsave {
            // 0. Here the CPU supports `XSAVE`.

            // 1. Detect `OSXSAVE`, that is, whether the OS is AVX enabled and
            // supports saving the state of the AVX/AVX2 vector registers on
            // context-switches, see:
            //
            // - [intel: is avx enabled?][is_avx_enabled],
            // - [mozilla: sse.cpp][mozilla_sse_cpp].
            //
            // [is_avx_enabled]: https://software.intel.com/en-us/blogs/2011/04/14/is-avx-enabled
            // [mozilla_sse_cpp]: https://hg.mozilla.org/mozilla-central/file/64bab5cbb9b6/mozglue/build/SSE.cpp#l190
            let cpu_osxsave = bit::test(proc_info_ecx as usize, 27);

            if cpu_osxsave {
                // 2. The OS must have signaled the CPU that it supports saving and
                // restoring the:
                //
                // * SSE -> `XCR0.SSE[1]`
                // * AVX -> `XCR0.AVX[2]`
                // * AVX-512 -> `XCR0.AVX-512[7:5]`.
                //
                // by setting the corresponding bits of `XCR0` to `1`.
                //
                // This is safe because the CPU supports `xsave`
                // and the OS has set `osxsave`.
                let xcr0 = unsafe { _xgetbv(0) };
                // Test `XCR0.SSE[1]` and `XCR0.AVX[2]` with the mask `0b110 == 6`:
                let os_avx_support = xcr0 & 6 == 6;
                // Test `XCR0.AVX-512[7:5]` with the mask `0b1110_0000 == 224`:
                let os_avx512_support = xcr0 & 224 == 224;

                // Only if the OS and the CPU support saving/restoring the AVX
                // registers we enable `xsave` support:
                if os_avx_support {
                    // See "13.3 ENABLING THE XSAVE FEATURE SET AND XSAVE-ENABLED
                    // FEATURES" in the "Intel® 64 and IA-32 Architectures Software
                    // Developer’s Manual, Volume 1: Basic Architecture":
                    //
                    // "Software enables the XSAVE feature set by setting
                    // CR4.OSXSAVE[bit 18] to 1 (e.g., with the MOV to CR4
                    // instruction). If this bit is 0, execution of any of XGETBV,
                    // XRSTOR, XRSTORS, XSAVE, XSAVEC, XSAVEOPT, XSAVES, and XSETBV
                    // causes an invalid-opcode exception (#UD)"
                    //
                    enable(proc_info_ecx, 26, Feature::xsave);

                    // For `xsaveopt`, `xsavec`, and `xsaves` we need to query:
                    // Processor Extended State Enumeration Sub-leaf (EAX = 0DH,
                    // ECX = 1):
                    if max_basic_leaf >= 0xd {
                        let CpuidResult {
                            eax: proc_extended_state1_eax,
                            ..
                        } = unsafe { __cpuid_count(0xd_u32, 1) };
                        enable(proc_extended_state1_eax, 0, Feature::xsaveopt);
                        enable(proc_extended_state1_eax, 1, Feature::xsavec);
                        enable(proc_extended_state1_eax, 3, Feature::xsaves);
                    }

                    // FMA (uses 256-bit wide registers):
                    enable(proc_info_ecx, 12, Feature::fma);

                    // And AVX/AVX2:
                    enable(proc_info_ecx, 28, Feature::avx);
                    enable(extended_features_ebx, 5, Feature::avx2);

                    // For AVX-512 the OS also needs to support saving/restoring
                    // the extended state, only then we enable AVX-512 support:
                    if os_avx512_support {
                        enable(extended_features_ebx, 16, Feature::avx512f);
                        enable(extended_features_ebx, 17, Feature::avx512dq);
                        enable(extended_features_ebx, 21, Feature::avx512_ifma);
                        enable(extended_features_ebx, 26, Feature::avx512pf);
                        enable(extended_features_ebx, 27, Feature::avx512er);
                        enable(extended_features_ebx, 28, Feature::avx512cd);
                        enable(extended_features_ebx, 30, Feature::avx512bw);
                        enable(extended_features_ebx, 31, Feature::avx512vl);
                        enable(extended_features_ecx, 1, Feature::avx512_vbmi);
                        enable(
                            extended_features_ecx,
                            14,
                            Feature::avx512_vpopcntdq,
                        );
                    }
                }
            }
        }

        // This detects ABM on AMD CPUs and LZCNT on Intel CPUs.
        // On intel CPUs with popcnt, lzcnt implements the
        // "missing part" of ABM, so we map both to the same
        // internal feature.
        //
        // The `is_x86_feature_detected!("lzcnt")` macro then
        // internally maps to Feature::abm.
        enable(extended_proc_info_ecx, 5, Feature::abm);
        // As Hygon Dhyana originates from AMD technology and shares most of the architecture with
        // AMD's family 17h, but with different CPU Vendor ID("HygonGenuine")/Family series
        // number(Family 18h).
        //
        // For CPUID feature bits, Hygon Dhyana(family 18h) share the same definition with AMD
        // family 17h.
        //
        // Related AMD CPUID specification is https://www.amd.com/system/files/TechDocs/25481.pdf.
        // Related Hygon kernel patch can be found on
        // http://lkml.kernel.org/r/5ce86123a7b9dad925ac583d88d2f921040e859b.1538583282.git.puwen@hygon.cn
        if vendor_id == *b"AuthenticAMD" || vendor_id == *b"HygonGenuine" {
            // These features are available on AMD arch CPUs:
            enable(extended_proc_info_ecx, 6, Feature::sse4a);
            enable(extended_proc_info_ecx, 21, Feature::tbm);
        }
    }

    value
}

#[cfg(test)]
mod tests {
    extern crate cupid;

    #[test]
    fn dump() {
        println!("aes: {:?}", is_x86_feature_detected!("aes"));
        println!("pclmulqdq: {:?}", is_x86_feature_detected!("pclmulqdq"));
        println!("rdrand: {:?}", is_x86_feature_detected!("rdrand"));
        println!("rdseed: {:?}", is_x86_feature_detected!("rdseed"));
        println!("tsc: {:?}", is_x86_feature_detected!("tsc"));
        println!("sse: {:?}", is_x86_feature_detected!("sse"));
        println!("sse2: {:?}", is_x86_feature_detected!("sse2"));
        println!("sse3: {:?}", is_x86_feature_detected!("sse3"));
        println!("ssse3: {:?}", is_x86_feature_detected!("ssse3"));
        println!("sse4.1: {:?}", is_x86_feature_detected!("sse4.1"));
        println!("sse4.2: {:?}", is_x86_feature_detected!("sse4.2"));
        println!("sse4a: {:?}", is_x86_feature_detected!("sse4a"));
        println!("sha: {:?}", is_x86_feature_detected!("sha"));
        println!("avx: {:?}", is_x86_feature_detected!("avx"));
        println!("avx2: {:?}", is_x86_feature_detected!("avx2"));
        println!("avx512f {:?}", is_x86_feature_detected!("avx512f"));
        println!("avx512cd {:?}", is_x86_feature_detected!("avx512cd"));
        println!("avx512er {:?}", is_x86_feature_detected!("avx512er"));
        println!("avx512pf {:?}", is_x86_feature_detected!("avx512pf"));
        println!("avx512bw {:?}", is_x86_feature_detected!("avx512bw"));
        println!("avx512dq {:?}", is_x86_feature_detected!("avx512dq"));
        println!("avx512vl {:?}", is_x86_feature_detected!("avx512vl"));
        println!("avx512_ifma {:?}", is_x86_feature_detected!("avx512ifma"));
        println!("avx512_vbmi {:?}", is_x86_feature_detected!("avx512vbmi"));
        println!(
            "avx512_vpopcntdq {:?}",
            is_x86_feature_detected!("avx512vpopcntdq")
        );
        println!("fma: {:?}", is_x86_feature_detected!("fma"));
        println!("abm: {:?}", is_x86_feature_detected!("abm"));
        println!("bmi: {:?}", is_x86_feature_detected!("bmi1"));
        println!("bmi2: {:?}", is_x86_feature_detected!("bmi2"));
        println!("tbm: {:?}", is_x86_feature_detected!("tbm"));
        println!("popcnt: {:?}", is_x86_feature_detected!("popcnt"));
        println!("lzcnt: {:?}", is_x86_feature_detected!("lzcnt"));
        println!("fxsr: {:?}", is_x86_feature_detected!("fxsr"));
        println!("xsave: {:?}", is_x86_feature_detected!("xsave"));
        println!("xsaveopt: {:?}", is_x86_feature_detected!("xsaveopt"));
        println!("xsaves: {:?}", is_x86_feature_detected!("xsaves"));
        println!("xsavec: {:?}", is_x86_feature_detected!("xsavec"));
        println!("cmpxchg16b: {:?}", is_x86_feature_detected!("cmpxchg16b"));
        println!("adx: {:?}", is_x86_feature_detected!("adx"));
        println!("rtm: {:?}", is_x86_feature_detected!("rtm"));
    }

    #[test]
    fn compare_with_cupid() {
        let information = cupid::master().unwrap();
        assert_eq!(is_x86_feature_detected!("aes"), information.aesni());
        assert_eq!(is_x86_feature_detected!("pclmulqdq"), information.pclmulqdq());
        assert_eq!(is_x86_feature_detected!("rdrand"), information.rdrand());
        assert_eq!(is_x86_feature_detected!("rdseed"), information.rdseed());
        assert_eq!(is_x86_feature_detected!("tsc"), information.tsc());
        assert_eq!(is_x86_feature_detected!("sse"), information.sse());
        assert_eq!(is_x86_feature_detected!("sse2"), information.sse2());
        assert_eq!(is_x86_feature_detected!("sse3"), information.sse3());
        assert_eq!(is_x86_feature_detected!("ssse3"), information.ssse3());
        assert_eq!(is_x86_feature_detected!("sse4.1"), information.sse4_1());
        assert_eq!(is_x86_feature_detected!("sse4.2"), information.sse4_2());
        assert_eq!(is_x86_feature_detected!("sse4a"), information.sse4a());
        assert_eq!(is_x86_feature_detected!("sha"), information.sha());
        assert_eq!(is_x86_feature_detected!("avx"), information.avx());
        assert_eq!(is_x86_feature_detected!("avx2"), information.avx2());
        assert_eq!(is_x86_feature_detected!("avx512f"), information.avx512f());
        assert_eq!(is_x86_feature_detected!("avx512cd"), information.avx512cd());
        assert_eq!(is_x86_feature_detected!("avx512er"), information.avx512er());
        assert_eq!(is_x86_feature_detected!("avx512pf"), information.avx512pf());
        assert_eq!(is_x86_feature_detected!("avx512bw"), information.avx512bw());
        assert_eq!(is_x86_feature_detected!("avx512dq"), information.avx512dq());
        assert_eq!(is_x86_feature_detected!("avx512vl"), information.avx512vl());
        assert_eq!(
            is_x86_feature_detected!("avx512ifma"),
            information.avx512_ifma()
        );
        assert_eq!(
            is_x86_feature_detected!("avx512vbmi"),
            information.avx512_vbmi()
        );
        assert_eq!(
            is_x86_feature_detected!("avx512vpopcntdq"),
            information.avx512_vpopcntdq()
        );
        assert_eq!(is_x86_feature_detected!("fma"), information.fma());
        assert_eq!(is_x86_feature_detected!("bmi1"), information.bmi1());
        assert_eq!(is_x86_feature_detected!("bmi2"), information.bmi2());
        assert_eq!(is_x86_feature_detected!("popcnt"), information.popcnt());
        assert_eq!(is_x86_feature_detected!("abm"), information.lzcnt());
        assert_eq!(is_x86_feature_detected!("tbm"), information.tbm());
        assert_eq!(is_x86_feature_detected!("lzcnt"), information.lzcnt());
        assert_eq!(is_x86_feature_detected!("xsave"), information.xsave());
        assert_eq!(is_x86_feature_detected!("xsaveopt"), information.xsaveopt());
        assert_eq!(
            is_x86_feature_detected!("xsavec"),
            information.xsavec_and_xrstor()
        );
        assert_eq!(
            is_x86_feature_detected!("xsaves"),
            information.xsaves_xrstors_and_ia32_xss()
        );
        assert_eq!(
            is_x86_feature_detected!("cmpxchg16b"),
            information.cmpxchg16b(),
        );
        assert_eq!(
            is_x86_feature_detected!("adx"),
            information.adx(),
        );
        assert_eq!(
            is_x86_feature_detected!("rtm"),
            information.rtm(),
        );
    }
}
