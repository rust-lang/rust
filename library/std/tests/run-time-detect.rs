//! These tests just check that the macros are available in std.

#![cfg_attr(
    any(
        all(target_arch = "arm", any(target_os = "linux", target_os = "android")),
        all(target_arch = "powerpc", target_os = "linux"),
        all(target_arch = "powerpc64", target_os = "linux"),
    ),
    feature(stdsimd)
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
    println!("dit: {}", is_aarch64_feature_detected!("dit"));
    println!("dotprod: {}", is_aarch64_feature_detected!("dotprod"));
    println!("dpb2: {}", is_aarch64_feature_detected!("dpb2"));
    println!("dpb: {}", is_aarch64_feature_detected!("dpb"));
    println!("f32mm: {}", is_aarch64_feature_detected!("f32mm"));
    println!("f64mm: {}", is_aarch64_feature_detected!("f64mm"));
    println!("fcma: {}", is_aarch64_feature_detected!("fcma"));
    println!("fhm: {}", is_aarch64_feature_detected!("fhm"));
    println!("flagm: {}", is_aarch64_feature_detected!("flagm"));
    println!("fp16: {}", is_aarch64_feature_detected!("fp16"));
    println!("frintts: {}", is_aarch64_feature_detected!("frintts"));
    println!("i8mm: {}", is_aarch64_feature_detected!("i8mm"));
    println!("jsconv: {}", is_aarch64_feature_detected!("jsconv"));
    println!("lse2: {}", is_aarch64_feature_detected!("lse2"));
    println!("lse: {}", is_aarch64_feature_detected!("lse"));
    println!("mte: {}", is_aarch64_feature_detected!("mte"));
    println!("neon: {}", is_aarch64_feature_detected!("neon"));
    println!("paca: {}", is_aarch64_feature_detected!("paca"));
    println!("pacg: {}", is_aarch64_feature_detected!("pacg"));
    println!("pmull: {}", is_aarch64_feature_detected!("pmull"));
    println!("rand: {}", is_aarch64_feature_detected!("rand"));
    println!("rcpc2: {}", is_aarch64_feature_detected!("rcpc2"));
    println!("rcpc: {}", is_aarch64_feature_detected!("rcpc"));
    println!("rdm: {}", is_aarch64_feature_detected!("rdm"));
    println!("sb: {}", is_aarch64_feature_detected!("sb"));
    println!("sha2: {}", is_aarch64_feature_detected!("sha2"));
    println!("sha3: {}", is_aarch64_feature_detected!("sha3"));
    println!("sm4: {}", is_aarch64_feature_detected!("sm4"));
    println!("ssbs: {}", is_aarch64_feature_detected!("ssbs"));
    println!("sve2-aes: {}", is_aarch64_feature_detected!("sve2-aes"));
    println!("sve2-bitperm: {}", is_aarch64_feature_detected!("sve2-bitperm"));
    println!("sve2-sha3: {}", is_aarch64_feature_detected!("sve2-sha3"));
    println!("sve2-sm4: {}", is_aarch64_feature_detected!("sve2-sm4"));
    println!("sve2: {}", is_aarch64_feature_detected!("sve2"));
    println!("sve: {}", is_aarch64_feature_detected!("sve"));
    println!("tme: {}", is_aarch64_feature_detected!("tme"));
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
    println!("avx512er: {:?}", is_x86_feature_detected!("avx512er"));
    println!("avx512f: {:?}", is_x86_feature_detected!("avx512f"));
    println!("avx512ifma: {:?}", is_x86_feature_detected!("avx512ifma"));
    println!("avx512pf: {:?}", is_x86_feature_detected!("avx512pf"));
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
