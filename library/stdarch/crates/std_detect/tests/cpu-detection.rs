#![allow(internal_features)]
#![feature(stdarch_internal)]
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_feature_detection))]
#![cfg_attr(
    any(target_arch = "aarch64", target_arch = "arm64ec"),
    feature(stdarch_aarch64_feature_detection)
)]
#![cfg_attr(target_arch = "powerpc", feature(stdarch_powerpc_feature_detection))]
#![cfg_attr(target_arch = "powerpc64", feature(stdarch_powerpc_feature_detection))]
#![cfg_attr(target_arch = "s390x", feature(stdarch_s390x_feature_detection))]
#![allow(clippy::unwrap_used, clippy::use_debug, clippy::print_stdout)]

#[cfg_attr(
    any(
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "arm64ec",
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "s390x",
    ),
    macro_use
)]
extern crate std_detect;

#[test]
fn all() {
    for (f, e) in std_detect::detect::features() {
        println!("{f}: {e}");
    }
}

#[test]
#[cfg(all(target_arch = "arm", target_os = "freebsd"))]
fn arm_freebsd() {
    println!("neon: {}", is_arm_feature_detected!("neon"));
    println!("pmull: {}", is_arm_feature_detected!("pmull"));
    println!("crc: {}", is_arm_feature_detected!("crc"));
    println!("aes: {}", is_arm_feature_detected!("aes"));
    println!("sha2: {}", is_arm_feature_detected!("sha2"));
}

#[test]
#[cfg(all(target_arch = "arm", any(target_os = "linux", target_os = "android")))]
fn arm_linux() {
    println!("neon: {}", is_arm_feature_detected!("neon"));
    println!("pmull: {}", is_arm_feature_detected!("pmull"));
    println!("crc: {}", is_arm_feature_detected!("crc"));
    println!("aes: {}", is_arm_feature_detected!("aes"));
    println!("sha2: {}", is_arm_feature_detected!("sha2"));
    println!("dotprod: {}", is_arm_feature_detected!("dotprod"));
    println!("i8mm: {}", is_arm_feature_detected!("i8mm"));
}

#[test]
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "android")
))]
fn aarch64_linux() {
    println!("asimd: {}", is_aarch64_feature_detected!("asimd"));
    println!("neon: {}", is_aarch64_feature_detected!("neon"));
    println!("pmull: {}", is_aarch64_feature_detected!("pmull"));
    println!("fp: {}", is_aarch64_feature_detected!("fp"));
    println!("fp16: {}", is_aarch64_feature_detected!("fp16"));
    println!("sve: {}", is_aarch64_feature_detected!("sve"));
    println!("crc: {}", is_aarch64_feature_detected!("crc"));
    println!("lse: {}", is_aarch64_feature_detected!("lse"));
    println!("lse2: {}", is_aarch64_feature_detected!("lse2"));
    println!("lse128: {}", is_aarch64_feature_detected!("lse128"));
    println!("rdm: {}", is_aarch64_feature_detected!("rdm"));
    println!("rcpc: {}", is_aarch64_feature_detected!("rcpc"));
    println!("rcpc2: {}", is_aarch64_feature_detected!("rcpc2"));
    println!("rcpc3: {}", is_aarch64_feature_detected!("rcpc3"));
    println!("dotprod: {}", is_aarch64_feature_detected!("dotprod"));
    println!("tme: {}", is_aarch64_feature_detected!("tme"));
    println!("fhm: {}", is_aarch64_feature_detected!("fhm"));
    println!("dit: {}", is_aarch64_feature_detected!("dit"));
    println!("flagm: {}", is_aarch64_feature_detected!("flagm"));
    println!("flagm2: {}", is_aarch64_feature_detected!("flagm2"));
    println!("ssbs: {}", is_aarch64_feature_detected!("ssbs"));
    println!("sb: {}", is_aarch64_feature_detected!("sb"));
    println!("paca: {}", is_aarch64_feature_detected!("paca"));
    println!("pacg: {}", is_aarch64_feature_detected!("pacg"));
    // println!("pauth-lr: {}", is_aarch64_feature_detected!("pauth-lr"));
    println!("dpb: {}", is_aarch64_feature_detected!("dpb"));
    println!("dpb2: {}", is_aarch64_feature_detected!("dpb2"));
    println!("sve-b16b16: {}", is_aarch64_feature_detected!("sve-b16b16"));
    println!("sve2: {}", is_aarch64_feature_detected!("sve2"));
    println!("sve2p1: {}", is_aarch64_feature_detected!("sve2p1"));
    println!("sve2-aes: {}", is_aarch64_feature_detected!("sve2-aes"));
    println!("sve2-sm4: {}", is_aarch64_feature_detected!("sve2-sm4"));
    println!("sve2-sha3: {}", is_aarch64_feature_detected!("sve2-sha3"));
    println!(
        "sve2-bitperm: {}",
        is_aarch64_feature_detected!("sve2-bitperm")
    );
    println!("frintts: {}", is_aarch64_feature_detected!("frintts"));
    println!("i8mm: {}", is_aarch64_feature_detected!("i8mm"));
    println!("f32mm: {}", is_aarch64_feature_detected!("f32mm"));
    println!("f64mm: {}", is_aarch64_feature_detected!("f64mm"));
    println!("bf16: {}", is_aarch64_feature_detected!("bf16"));
    println!("rand: {}", is_aarch64_feature_detected!("rand"));
    println!("bti: {}", is_aarch64_feature_detected!("bti"));
    println!("mte: {}", is_aarch64_feature_detected!("mte"));
    println!("jsconv: {}", is_aarch64_feature_detected!("jsconv"));
    println!("fcma: {}", is_aarch64_feature_detected!("fcma"));
    println!("aes: {}", is_aarch64_feature_detected!("aes"));
    println!("sha2: {}", is_aarch64_feature_detected!("sha2"));
    println!("sha3: {}", is_aarch64_feature_detected!("sha3"));
    println!("sm4: {}", is_aarch64_feature_detected!("sm4"));
    println!("hbc: {}", is_aarch64_feature_detected!("hbc"));
    println!("mops: {}", is_aarch64_feature_detected!("mops"));
    println!("ecv: {}", is_aarch64_feature_detected!("ecv"));
    println!("cssc: {}", is_aarch64_feature_detected!("cssc"));
    println!("fpmr: {}", is_aarch64_feature_detected!("fpmr"));
    println!("lut: {}", is_aarch64_feature_detected!("lut"));
    println!("faminmax: {}", is_aarch64_feature_detected!("faminmax"));
    println!("fp8: {}", is_aarch64_feature_detected!("fp8"));
    println!("fp8fma: {}", is_aarch64_feature_detected!("fp8fma"));
    println!("fp8dot4: {}", is_aarch64_feature_detected!("fp8dot4"));
    println!("fp8dot2: {}", is_aarch64_feature_detected!("fp8dot2"));
    println!("wfxt: {}", is_aarch64_feature_detected!("wfxt"));
    println!("sme: {}", is_aarch64_feature_detected!("sme"));
    println!("sme-b16b16: {}", is_aarch64_feature_detected!("sme-b16b16"));
    println!("sme-i16i64: {}", is_aarch64_feature_detected!("sme-i16i64"));
    println!("sme-f64f64: {}", is_aarch64_feature_detected!("sme-f64f64"));
    println!("sme-fa64: {}", is_aarch64_feature_detected!("sme-fa64"));
    println!("sme2: {}", is_aarch64_feature_detected!("sme2"));
    println!("sme2p1: {}", is_aarch64_feature_detected!("sme2p1"));
    println!("sme-f16f16: {}", is_aarch64_feature_detected!("sme-f16f16"));
    println!("sme-lutv2: {}", is_aarch64_feature_detected!("sme-lutv2"));
    println!("sme-f8f16: {}", is_aarch64_feature_detected!("sme-f8f16"));
    println!("sme-f8f32: {}", is_aarch64_feature_detected!("sme-f8f32"));
    println!(
        "ssve-fp8fma: {}",
        is_aarch64_feature_detected!("ssve-fp8fma")
    );
    println!(
        "ssve-fp8dot4: {}",
        is_aarch64_feature_detected!("ssve-fp8dot4")
    );
    println!(
        "ssve-fp8dot2: {}",
        is_aarch64_feature_detected!("ssve-fp8dot2")
    );
}

#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm64ec"),
    target_os = "windows"
))]
fn aarch64_windows() {
    println!("asimd: {:?}", is_aarch64_feature_detected!("asimd"));
    println!("fp: {:?}", is_aarch64_feature_detected!("fp"));
    println!("crc: {:?}", is_aarch64_feature_detected!("crc"));
    println!("lse: {:?}", is_aarch64_feature_detected!("lse"));
    println!("dotprod: {:?}", is_aarch64_feature_detected!("dotprod"));
    println!("jsconv: {:?}", is_aarch64_feature_detected!("jsconv"));
    println!("rcpc: {:?}", is_aarch64_feature_detected!("rcpc"));
    println!("aes: {:?}", is_aarch64_feature_detected!("aes"));
    println!("pmull: {:?}", is_aarch64_feature_detected!("pmull"));
    println!("sha2: {:?}", is_aarch64_feature_detected!("sha2"));
}

#[test]
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "freebsd", target_os = "openbsd")
))]
fn aarch64_bsd() {
    println!("asimd: {:?}", is_aarch64_feature_detected!("asimd"));
    println!("pmull: {:?}", is_aarch64_feature_detected!("pmull"));
    println!("fp: {:?}", is_aarch64_feature_detected!("fp"));
    println!("fp16: {:?}", is_aarch64_feature_detected!("fp16"));
    println!("sve: {:?}", is_aarch64_feature_detected!("sve"));
    println!("crc: {:?}", is_aarch64_feature_detected!("crc"));
    println!("lse: {:?}", is_aarch64_feature_detected!("lse"));
    println!("lse2: {:?}", is_aarch64_feature_detected!("lse2"));
    println!("rdm: {:?}", is_aarch64_feature_detected!("rdm"));
    println!("rcpc: {:?}", is_aarch64_feature_detected!("rcpc"));
    println!("dotprod: {:?}", is_aarch64_feature_detected!("dotprod"));
    println!("tme: {:?}", is_aarch64_feature_detected!("tme"));
    println!("paca: {:?}", is_aarch64_feature_detected!("paca"));
    println!("pacg: {:?}", is_aarch64_feature_detected!("pacg"));
    println!("aes: {:?}", is_aarch64_feature_detected!("aes"));
    println!("sha2: {:?}", is_aarch64_feature_detected!("sha2"));
}

#[test]
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
fn aarch64_darwin() {
    println!("asimd: {:?}", is_aarch64_feature_detected!("asimd"));
    println!("fp: {:?}", is_aarch64_feature_detected!("fp"));
    println!("fp16: {:?}", is_aarch64_feature_detected!("fp16"));
    println!("pmull: {:?}", is_aarch64_feature_detected!("pmull"));
    println!("crc: {:?}", is_aarch64_feature_detected!("crc"));
    println!("lse: {:?}", is_aarch64_feature_detected!("lse"));
    println!("lse2: {:?}", is_aarch64_feature_detected!("lse2"));
    println!("rdm: {:?}", is_aarch64_feature_detected!("rdm"));
    println!("rcpc: {:?}", is_aarch64_feature_detected!("rcpc"));
    println!("rcpc2: {:?}", is_aarch64_feature_detected!("rcpc2"));
    println!("dotprod: {:?}", is_aarch64_feature_detected!("dotprod"));
    println!("fhm: {:?}", is_aarch64_feature_detected!("fhm"));
    println!("flagm: {:?}", is_aarch64_feature_detected!("flagm"));
    println!("ssbs: {:?}", is_aarch64_feature_detected!("ssbs"));
    println!("sb: {:?}", is_aarch64_feature_detected!("sb"));
    println!("paca: {:?}", is_aarch64_feature_detected!("paca"));
    println!("dpb: {:?}", is_aarch64_feature_detected!("dpb"));
    println!("dpb2: {:?}", is_aarch64_feature_detected!("dpb2"));
    println!("frintts: {:?}", is_aarch64_feature_detected!("frintts"));
    println!("i8mm: {:?}", is_aarch64_feature_detected!("i8mm"));
    println!("bf16: {:?}", is_aarch64_feature_detected!("bf16"));
    println!("bti: {:?}", is_aarch64_feature_detected!("bti"));
    println!("fcma: {:?}", is_aarch64_feature_detected!("fcma"));
    println!("jsconv: {:?}", is_aarch64_feature_detected!("jsconv"));
    println!("aes: {:?}", is_aarch64_feature_detected!("aes"));
    println!("sha2: {:?}", is_aarch64_feature_detected!("sha2"));
    println!("sha3: {:?}", is_aarch64_feature_detected!("sha3"));
}

#[test]
#[cfg(all(target_arch = "powerpc", target_os = "linux"))]
fn powerpc_linux() {
    println!("altivec: {}", is_powerpc_feature_detected!("altivec"));
    println!("vsx: {}", is_powerpc_feature_detected!("vsx"));
    println!("power8: {}", is_powerpc_feature_detected!("power8"));
}

#[test]
#[cfg(all(
    target_arch = "powerpc64",
    any(target_os = "linux", target_os = "freebsd"),
))]
fn powerpc64_linux_or_freebsd() {
    println!("altivec: {}", is_powerpc64_feature_detected!("altivec"));
    println!("vsx: {}", is_powerpc64_feature_detected!("vsx"));
    println!("power8: {}", is_powerpc64_feature_detected!("power8"));
}

#[test]
#[cfg(all(target_arch = "s390x", target_os = "linux",))]
fn s390x_linux() {
    println!("vector: {}", is_s390x_feature_detected!("vector"));
}
