#![feature(cfg_target_feature)]
#![cfg_attr(feature = "strict", deny(warnings))]
#![cfg_attr(feature = "cargo-clippy", allow(option_unwrap_used))]

extern crate cupid;

#[macro_use]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
extern crate stdsimd;

#[test]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn works() {
    let information = cupid::master().unwrap();
    assert_eq!(cfg_feature_enabled!("sse"), information.sse());
    assert_eq!(cfg_feature_enabled!("sse2"), information.sse2());
    assert_eq!(cfg_feature_enabled!("sse3"), information.sse3());
    assert_eq!(cfg_feature_enabled!("ssse3"), information.ssse3());
    assert_eq!(cfg_feature_enabled!("sse4.1"), information.sse4_1());
    assert_eq!(cfg_feature_enabled!("sse4.2"), information.sse4_2());
    assert_eq!(cfg_feature_enabled!("avx"), information.avx());
    assert_eq!(cfg_feature_enabled!("avx2"), information.avx2());
    assert_eq!(cfg_feature_enabled!("fma"), information.fma());
    assert_eq!(cfg_feature_enabled!("bmi"), information.bmi1());
    assert_eq!(cfg_feature_enabled!("bmi2"), information.bmi2());
    assert_eq!(cfg_feature_enabled!("popcnt"), information.popcnt());

    // TODO: tbm, abm, lzcnt
}
