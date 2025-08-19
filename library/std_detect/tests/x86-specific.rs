#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![allow(internal_features)]
#![feature(stdarch_internal, x86_amx_intrinsics, xop_target_feature, movrs_target_feature)]

#[macro_use]
extern crate std_detect;

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
    println!("f16c: {:?}", is_x86_feature_detected!("f16c"));
    println!("avx: {:?}", is_x86_feature_detected!("avx"));
    println!("avx2: {:?}", is_x86_feature_detected!("avx2"));
    println!("sha512: {:?}", is_x86_feature_detected!("sha512"));
    println!("sm3: {:?}", is_x86_feature_detected!("sm3"));
    println!("sm4: {:?}", is_x86_feature_detected!("sm4"));
    println!("avx512f: {:?}", is_x86_feature_detected!("avx512f"));
    println!("avx512cd: {:?}", is_x86_feature_detected!("avx512cd"));
    println!("avx512er: {:?}", is_x86_feature_detected!("avx512er"));
    println!("avx512pf: {:?}", is_x86_feature_detected!("avx512pf"));
    println!("avx512bw: {:?}", is_x86_feature_detected!("avx512bw"));
    println!("avx512dq: {:?}", is_x86_feature_detected!("avx512dq"));
    println!("avx512vl: {:?}", is_x86_feature_detected!("avx512vl"));
    println!("avx512_ifma: {:?}", is_x86_feature_detected!("avx512ifma"));
    println!("avx512vbmi {:?}", is_x86_feature_detected!("avx512vbmi"));
    println!("avx512_vpopcntdq: {:?}", is_x86_feature_detected!("avx512vpopcntdq"));
    println!("avx512vbmi2: {:?}", is_x86_feature_detected!("avx512vbmi2"));
    println!("gfni: {:?}", is_x86_feature_detected!("gfni"));
    println!("vaes: {:?}", is_x86_feature_detected!("vaes"));
    println!("vpclmulqdq: {:?}", is_x86_feature_detected!("vpclmulqdq"));
    println!("avx512vnni: {:?}", is_x86_feature_detected!("avx512vnni"));
    println!("avx512bitalg: {:?}", is_x86_feature_detected!("avx512bitalg"));
    println!("avx512bf16: {:?}", is_x86_feature_detected!("avx512bf16"));
    println!("avx512vp2intersect: {:?}", is_x86_feature_detected!("avx512vp2intersect"));
    println!("avx512fp16: {:?}", is_x86_feature_detected!("avx512fp16"));
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
    println!("movbe: {:?}", is_x86_feature_detected!("movbe"));
    println!("avxvnni: {:?}", is_x86_feature_detected!("avxvnni"));
    println!("avxvnniint8: {:?}", is_x86_feature_detected!("avxvnniint8"));
    println!("avxneconvert: {:?}", is_x86_feature_detected!("avxneconvert"));
    println!("avxifma: {:?}", is_x86_feature_detected!("avxifma"));
    println!("avxvnniint16: {:?}", is_x86_feature_detected!("avxvnniint16"));
    println!("amx-bf16: {:?}", is_x86_feature_detected!("amx-bf16"));
    println!("amx-tile: {:?}", is_x86_feature_detected!("amx-tile"));
    println!("amx-int8: {:?}", is_x86_feature_detected!("amx-int8"));
    println!("amx-fp16: {:?}", is_x86_feature_detected!("amx-fp16"));
    println!("amx-complex: {:?}", is_x86_feature_detected!("amx-complex"));
    println!("xop: {:?}", is_x86_feature_detected!("xop"));
    println!("kl: {:?}", is_x86_feature_detected!("kl"));
    println!("widekl: {:?}", is_x86_feature_detected!("widekl"));
    println!("movrs: {:?}", is_x86_feature_detected!("movrs"));
    println!("amx-fp8: {:?}", is_x86_feature_detected!("amx-fp8"));
    println!("amx-transpose: {:?}", is_x86_feature_detected!("amx-transpose"));
    println!("amx-tf32: {:?}", is_x86_feature_detected!("amx-tf32"));
    println!("amx-avx512: {:?}", is_x86_feature_detected!("amx-avx512"));
    println!("amx-movrs: {:?}", is_x86_feature_detected!("amx-movrs"));
}

#[test]
#[allow(deprecated)]
fn x86_deprecated() {
    println!("avx512gfni {:?}", is_x86_feature_detected!("avx512gfni"));
    println!("avx512vaes {:?}", is_x86_feature_detected!("avx512vaes"));
    println!("avx512vpclmulqdq {:?}", is_x86_feature_detected!("avx512vpclmulqdq"));
}
