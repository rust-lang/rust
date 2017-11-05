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
    // assert_eq!(cfg_feature_enabled!("avx512f"), information.avx512f());
    // assert_eq!(cfg_feature_enabled!("avx512cd"), information.avx512cd());
    // assert_eq!(cfg_feature_enabled!("avx512er"), information.avx512er());
    // assert_eq!(cfg_feature_enabled!("avx512pf"), information.avx512pf());
    // assert_eq!(cfg_feature_enabled!("avx512bw"), information.avx512bw());
    // assert_eq!(cfg_feature_enabled!("avx512dq"), information.avx512dq());
    // assert_eq!(cfg_feature_enabled!("avx512vl"), information.avx512vl());
    // assert_eq!(cfg_feature_enabled!("avx512ifma"), information.avx512ifma());
    // assert_eq!(cfg_feature_enabled!("avx512vbmi"), information.avx512vbmi());
    // assert_eq!(cfg_feature_enabled!("avx512vpopcntdq"), information.avx512vpopcntdq());
    assert_eq!(cfg_feature_enabled!("fma"), information.fma());
    assert_eq!(cfg_feature_enabled!("bmi"), information.bmi1());
    assert_eq!(cfg_feature_enabled!("bmi2"), information.bmi2());
    assert_eq!(cfg_feature_enabled!("popcnt"), information.popcnt());
    // assert_eq!(cfg_feature_enabled!("sse4a"), information.sse4a());
    assert_eq!(cfg_feature_enabled!("abm"), information.lzcnt());
    assert_eq!(cfg_feature_enabled!("tbm"), information.tbm());
    assert_eq!(cfg_feature_enabled!("lzcnt"), information.lzcnt());
    assert_eq!(cfg_feature_enabled!("xsave"), information.xsave());
    assert_eq!(cfg_feature_enabled!("xsaveopt"), information.xsaveopt());
    assert_eq!(cfg_feature_enabled!("xsavec"), information.xsavec_and_xrstor());
    assert_eq!(cfg_feature_enabled!("xsavec"), information.xsaves_xrstors_and_ia32_xss());
}
