//! These tests just check that the macros are available in std.

#![cfg_attr(
    all(target_arch = "arm", any(target_os = "linux", target_os = "android")),
    feature(stdarch_arm_feature_detection)
)]
#![cfg_attr(
    all(target_arch = "aarch64", any(target_os = "linux", target_os = "android")),
    feature(stdarch_aarch64_feature_detection)
)]
#![cfg_attr(
    all(target_arch = "s390x", target_os = "linux"),
    feature(stdarch_s390x_feature_detection)
)]
#![cfg_attr(
    all(target_arch = "powerpc", target_os = "linux"),
    feature(stdarch_powerpc_feature_detection)
)]
#![cfg_attr(
    all(target_arch = "powerpc64", target_os = "linux"),
    feature(stdarch_powerpc_feature_detection)
)]

#[test]
#[cfg(all(target_arch = "arm", any(target_os = "linux", target_os = "android")))]
fn arm_linux() {
    use std::arch::is_arm_feature_detected;
    // tidy-alphabetical-start
    println!("aes: {}", is_arm_feature_detected!("aes"));
    println!("crc: {}", is_arm_feature_detected!("crc"));
    println!("neon: {}", is_arm_feature_detected!("neon"));
    println!("pmull: {}", is_arm_feature_detected!("pmull"));
    println!("sha2: {}", is_arm_feature_detected!("sha2"));
    // tidy-alphabetical-end
}

#[test]
#[cfg(all(target_arch = "aarch64", any(target_os = "linux", target_os = "android")))]
fn aarch64_linux() {
    use std::arch::is_aarch64_feature_detected;
    // tidy-alphabetical-start
    println!("aes: {}", is_aarch64_feature_detected!("aes"));
    println!("asimd: {}", is_aarch64_feature_detected!("asimd"));
    println!("bf16: {}", is_aarch64_feature_detected!("bf16"));
    println!("bti: {}", is_aarch64_feature_detected!("bti"));
    println!("crc: {}", is_aarch64_feature_detected!("crc"));
    println!("cssc: {}", is_aarch64_feature_detected!("cssc"));
    println!("dit: {}", is_aarch64_feature_detected!("dit"));
    println!("dotprod: {}", is_aarch64_feature_detected!("dotprod"));
    println!("dpb2: {}", is_aarch64_feature_detected!("dpb2"));
    println!("dpb: {}", is_aarch64_feature_detected!("dpb"));
    println!("ecv: {}", is_aarch64_feature_detected!("ecv"));
    println!("f32mm: {}", is_aarch64_feature_detected!("f32mm"));
    println!("f64mm: {}", is_aarch64_feature_detected!("f64mm"));
    println!("faminmax: {}", is_aarch64_feature_detected!("faminmax"));
    println!("fcma: {}", is_aarch64_feature_detected!("fcma"));
    println!("fhm: {}", is_aarch64_feature_detected!("fhm"));
    println!("flagm2: {}", is_aarch64_feature_detected!("flagm2"));
    println!("flagm: {}", is_aarch64_feature_detected!("flagm"));
    println!("fp8: {}", is_aarch64_feature_detected!("fp8"));
    println!("fp8dot2: {}", is_aarch64_feature_detected!("fp8dot2"));
    println!("fp8dot4: {}", is_aarch64_feature_detected!("fp8dot4"));
    println!("fp8fma: {}", is_aarch64_feature_detected!("fp8fma"));
    println!("fp16: {}", is_aarch64_feature_detected!("fp16"));
    println!("fpmr: {}", is_aarch64_feature_detected!("fpmr"));
    println!("frintts: {}", is_aarch64_feature_detected!("frintts"));
    println!("hbc: {}", is_aarch64_feature_detected!("hbc"));
    println!("i8mm: {}", is_aarch64_feature_detected!("i8mm"));
    println!("jsconv: {}", is_aarch64_feature_detected!("jsconv"));
    println!("lse2: {}", is_aarch64_feature_detected!("lse2"));
    println!("lse128: {}", is_aarch64_feature_detected!("lse128"));
    println!("lse: {}", is_aarch64_feature_detected!("lse"));
    println!("lut: {}", is_aarch64_feature_detected!("lut"));
    println!("mops: {}", is_aarch64_feature_detected!("mops"));
    println!("mte: {}", is_aarch64_feature_detected!("mte"));
    println!("neon: {}", is_aarch64_feature_detected!("neon"));
    println!("paca: {}", is_aarch64_feature_detected!("paca"));
    println!("pacg: {}", is_aarch64_feature_detected!("pacg"));
    println!("pmull: {}", is_aarch64_feature_detected!("pmull"));
    println!("rand: {}", is_aarch64_feature_detected!("rand"));
    println!("rcpc2: {}", is_aarch64_feature_detected!("rcpc2"));
    println!("rcpc3: {}", is_aarch64_feature_detected!("rcpc3"));
    println!("rcpc: {}", is_aarch64_feature_detected!("rcpc"));
    println!("rdm: {}", is_aarch64_feature_detected!("rdm"));
    println!("sb: {}", is_aarch64_feature_detected!("sb"));
    println!("sha2: {}", is_aarch64_feature_detected!("sha2"));
    println!("sha3: {}", is_aarch64_feature_detected!("sha3"));
    println!("sm4: {}", is_aarch64_feature_detected!("sm4"));
    println!("sme-b16b16: {}", is_aarch64_feature_detected!("sme-b16b16"));
    println!("sme-f8f16: {}", is_aarch64_feature_detected!("sme-f8f16"));
    println!("sme-f8f32: {}", is_aarch64_feature_detected!("sme-f8f32"));
    println!("sme-f16f16: {}", is_aarch64_feature_detected!("sme-f16f16"));
    println!("sme-f64f64: {}", is_aarch64_feature_detected!("sme-f64f64"));
    println!("sme-fa64: {}", is_aarch64_feature_detected!("sme-fa64"));
    println!("sme-i16i64: {}", is_aarch64_feature_detected!("sme-i16i64"));
    println!("sme-lutv2: {}", is_aarch64_feature_detected!("sme-lutv2"));
    println!("sme2: {}", is_aarch64_feature_detected!("sme2"));
    println!("sme2p1: {}", is_aarch64_feature_detected!("sme2p1"));
    println!("sme: {}", is_aarch64_feature_detected!("sme"));
    println!("ssbs: {}", is_aarch64_feature_detected!("ssbs"));
    println!("ssve-fp8dot2: {}", is_aarch64_feature_detected!("ssve-fp8dot2"));
    println!("ssve-fp8dot4: {}", is_aarch64_feature_detected!("ssve-fp8dot4"));
    println!("ssve-fp8fma: {}", is_aarch64_feature_detected!("ssve-fp8fma"));
    println!("sve-b16b16: {}", is_aarch64_feature_detected!("sve-b16b16"));
    println!("sve2-aes: {}", is_aarch64_feature_detected!("sve2-aes"));
    println!("sve2-bitperm: {}", is_aarch64_feature_detected!("sve2-bitperm"));
    println!("sve2-sha3: {}", is_aarch64_feature_detected!("sve2-sha3"));
    println!("sve2-sm4: {}", is_aarch64_feature_detected!("sve2-sm4"));
    println!("sve2: {}", is_aarch64_feature_detected!("sve2"));
    println!("sve2p1: {}", is_aarch64_feature_detected!("sve2p1"));
    println!("sve: {}", is_aarch64_feature_detected!("sve"));
    println!("tme: {}", is_aarch64_feature_detected!("tme"));
    println!("wfxt: {}", is_aarch64_feature_detected!("wfxt"));
    // tidy-alphabetical-end
}

#[test]
#[cfg(all(target_arch = "powerpc", target_os = "linux"))]
fn powerpc_linux() {
    use std::arch::is_powerpc_feature_detected;
    // tidy-alphabetical-start
    println!("altivec: {}", is_powerpc_feature_detected!("altivec"));
    println!("power8: {}", is_powerpc_feature_detected!("power8"));
    println!("vsx: {}", is_powerpc_feature_detected!("vsx"));
    // tidy-alphabetical-end
}

#[test]
#[cfg(all(target_arch = "powerpc64", target_os = "linux"))]
fn powerpc64_linux() {
    use std::arch::is_powerpc64_feature_detected;
    // tidy-alphabetical-start
    println!("altivec: {}", is_powerpc64_feature_detected!("altivec"));
    println!("power8: {}", is_powerpc64_feature_detected!("power8"));
    println!("vsx: {}", is_powerpc64_feature_detected!("vsx"));
    // tidy-alphabetical-end
}

#[test]
#[cfg(all(target_arch = "s390x", target_os = "linux"))]
fn s390x_linux() {
    use std::arch::is_s390x_feature_detected;
    // tidy-alphabetical-start
    println!("deflate-conversion: {}", is_s390x_feature_detected!("deflate-conversion"));
    println!("enhanced-sort: {}", is_s390x_feature_detected!("enhanced-sort"));
    println!("guarded-storage: {}", is_s390x_feature_detected!("guarded-storage"));
    println!("high-word: {}", is_s390x_feature_detected!("high-word"));
    println!("nnp-assist: {}", is_s390x_feature_detected!("nnp-assist"));
    println!("transactional-execution: {}", is_s390x_feature_detected!("transactional-execution"));
    println!("vector-enhancements-1: {}", is_s390x_feature_detected!("vector-enhancements-1"));
    println!("vector-enhancements-2: {}", is_s390x_feature_detected!("vector-enhancements-2"));
    println!(
        "vector-packed-decimal-enhancement-2: {}",
        is_s390x_feature_detected!("vector-packed-decimal-enhancement-2")
    );
    println!(
        "vector-packed-decimal-enhancement: {}",
        is_s390x_feature_detected!("vector-packed-decimal-enhancement")
    );
    println!("vector-packed-decimal: {}", is_s390x_feature_detected!("vector-packed-decimal"));
    println!("vector: {}", is_s390x_feature_detected!("vector"));
    // tidy-alphabetical-end
}

#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_all() {
    use std::arch::is_x86_feature_detected;

    // the below is the set of features we can test at runtime, but don't actually
    // use to gate anything and are thus not part of the X86_ALLOWED_FEATURES list

    println!("abm: {:?}", is_x86_feature_detected!("abm")); // this is a synonym for lzcnt but we test it anyways
    println!("mmx: {:?}", is_x86_feature_detected!("mmx"));
    println!("tsc: {:?}", is_x86_feature_detected!("tsc"));

    // the below is in alphabetical order and matches
    // the order of X86_ALLOWED_FEATURES in rustc_codegen_ssa's target_features.rs

    // tidy-alphabetical-start
    println!("adx: {:?}", is_x86_feature_detected!("adx"));
    println!("aes: {:?}", is_x86_feature_detected!("aes"));
    println!("avx2: {:?}", is_x86_feature_detected!("avx2"));
    println!("avx512bf16: {:?}", is_x86_feature_detected!("avx512bf16"));
    println!("avx512bitalg: {:?}", is_x86_feature_detected!("avx512bitalg"));
    println!("avx512bw: {:?}", is_x86_feature_detected!("avx512bw"));
    println!("avx512cd: {:?}", is_x86_feature_detected!("avx512cd"));
    println!("avx512dq: {:?}", is_x86_feature_detected!("avx512dq"));
    println!("avx512f: {:?}", is_x86_feature_detected!("avx512f"));
    println!("avx512ifma: {:?}", is_x86_feature_detected!("avx512ifma"));
    println!("avx512vbmi2: {:?}", is_x86_feature_detected!("avx512vbmi2"));
    println!("avx512vbmi: {:?}", is_x86_feature_detected!("avx512vbmi"));
    println!("avx512vl: {:?}", is_x86_feature_detected!("avx512vl"));
    println!("avx512vnni: {:?}", is_x86_feature_detected!("avx512vnni"));
    println!("avx512vp2intersect: {:?}", is_x86_feature_detected!("avx512vp2intersect"));
    println!("avx512vpopcntdq: {:?}", is_x86_feature_detected!("avx512vpopcntdq"));
    println!("avx: {:?}", is_x86_feature_detected!("avx"));
    println!("bmi1: {:?}", is_x86_feature_detected!("bmi1"));
    println!("bmi2: {:?}", is_x86_feature_detected!("bmi2"));
    println!("cmpxchg16b: {:?}", is_x86_feature_detected!("cmpxchg16b"));
    println!("f16c: {:?}", is_x86_feature_detected!("f16c"));
    println!("fma: {:?}", is_x86_feature_detected!("fma"));
    println!("fxsr: {:?}", is_x86_feature_detected!("fxsr"));
    println!("gfni: {:?}", is_x86_feature_detected!("gfni"));
    println!("lzcnt: {:?}", is_x86_feature_detected!("lzcnt"));
    //println!("movbe: {:?}", is_x86_feature_detected!("movbe")); // movbe is unsupported as a target feature
    println!("pclmulqdq: {:?}", is_x86_feature_detected!("pclmulqdq"));
    println!("popcnt: {:?}", is_x86_feature_detected!("popcnt"));
    println!("rdrand: {:?}", is_x86_feature_detected!("rdrand"));
    println!("rdseed: {:?}", is_x86_feature_detected!("rdseed"));
    println!("rtm: {:?}", is_x86_feature_detected!("rtm"));
    println!("sha: {:?}", is_x86_feature_detected!("sha"));
    println!("sse2: {:?}", is_x86_feature_detected!("sse2"));
    println!("sse3: {:?}", is_x86_feature_detected!("sse3"));
    println!("sse4.1: {:?}", is_x86_feature_detected!("sse4.1"));
    println!("sse4.2: {:?}", is_x86_feature_detected!("sse4.2"));
    println!("sse4a: {:?}", is_x86_feature_detected!("sse4a"));
    println!("sse: {:?}", is_x86_feature_detected!("sse"));
    println!("ssse3: {:?}", is_x86_feature_detected!("ssse3"));
    println!("tbm: {:?}", is_x86_feature_detected!("tbm"));
    println!("vaes: {:?}", is_x86_feature_detected!("vaes"));
    println!("vpclmulqdq: {:?}", is_x86_feature_detected!("vpclmulqdq"));
    println!("xsave: {:?}", is_x86_feature_detected!("xsave"));
    println!("xsavec: {:?}", is_x86_feature_detected!("xsavec"));
    println!("xsaveopt: {:?}", is_x86_feature_detected!("xsaveopt"));
    println!("xsaves: {:?}", is_x86_feature_detected!("xsaves"));
    // tidy-alphabetical-end
}
