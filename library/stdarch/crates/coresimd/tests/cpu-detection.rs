#![feature(cfg_target_feature, stdsimd)]
#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "cargo-clippy",
            allow(option_unwrap_used, print_stdout, use_debug))]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_use]
extern crate stdsimd;

#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_all() {
    println!("sse: {:?}", is_target_feature_detected!("sse"));
    println!("sse2: {:?}", is_target_feature_detected!("sse2"));
    println!("sse3: {:?}", is_target_feature_detected!("sse3"));
    println!("ssse3: {:?}", is_target_feature_detected!("ssse3"));
    println!("sse4.1: {:?}", is_target_feature_detected!("sse4.1"));
    println!("sse4.2: {:?}", is_target_feature_detected!("sse4.2"));
    println!("sse4a: {:?}", is_target_feature_detected!("sse4a"));
    println!("avx: {:?}", is_target_feature_detected!("avx"));
    println!("avx2: {:?}", is_target_feature_detected!("avx2"));
    println!("avx512f {:?}", is_target_feature_detected!("avx512f"));
    println!("avx512cd {:?}", is_target_feature_detected!("avx512cd"));
    println!("avx512er {:?}", is_target_feature_detected!("avx512er"));
    println!("avx512pf {:?}", is_target_feature_detected!("avx512pf"));
    println!("avx512bw {:?}", is_target_feature_detected!("avx512bw"));
    println!("avx512dq {:?}", is_target_feature_detected!("avx512dq"));
    println!("avx512vl {:?}", is_target_feature_detected!("avx512vl"));
    println!(
        "avx512_ifma {:?}",
        is_target_feature_detected!("avx512ifma")
    );
    println!(
        "avx512_vbmi {:?}",
        is_target_feature_detected!("avx512vbmi")
    );
    println!(
        "avx512_vpopcntdq {:?}",
        is_target_feature_detected!("avx512vpopcntdq")
    );
    println!("fma: {:?}", is_target_feature_detected!("fma"));
    println!("abm: {:?}", is_target_feature_detected!("abm"));
    println!("bmi: {:?}", is_target_feature_detected!("bmi1"));
    println!("bmi2: {:?}", is_target_feature_detected!("bmi2"));
    println!("tbm: {:?}", is_target_feature_detected!("tbm"));
    println!("popcnt: {:?}", is_target_feature_detected!("popcnt"));
    println!("lzcnt: {:?}", is_target_feature_detected!("lzcnt"));
    println!("fxsr: {:?}", is_target_feature_detected!("fxsr"));
    println!("xsave: {:?}", is_target_feature_detected!("xsave"));
    println!("xsaveopt: {:?}", is_target_feature_detected!("xsaveopt"));
    println!("xsaves: {:?}", is_target_feature_detected!("xsaves"));
    println!("xsavec: {:?}", is_target_feature_detected!("xsavec"));
}
