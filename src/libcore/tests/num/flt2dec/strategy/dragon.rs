use std::prelude::v1::*;
use super::super::*;
use core::num::bignum::Big32x40 as Big;
use core::num::flt2dec::strategy::dragon::*;

#[test]
fn test_mul_pow10() {
    let mut prevpow10 = Big::from_small(1);
    for i in 1..340 {
        let mut curpow10 = Big::from_small(1);
        mul_pow10(&mut curpow10, i);
        assert_eq!(curpow10, *prevpow10.clone().mul_small(10));
        prevpow10 = curpow10;
    }
}

#[cfg_attr(all(target_arch = "wasm32", target_os = "emscripten"), ignore)] // issue 42630
#[test]
fn shortest_sanity_test() {
    f64_shortest_sanity_test(format_shortest);
    f32_shortest_sanity_test(format_shortest);
    more_shortest_sanity_test(format_shortest);
}

#[test]
fn exact_sanity_test() {
    // This test ends up running what I can only assume is some corner-ish case
    // of the `exp2` library function, defined in whatever C runtime we're
    // using. In VS 2013 this function apparently had a bug as this test fails
    // when linked, but with VS 2015 the bug appears fixed as the test runs just
    // fine.
    //
    // The bug seems to be a difference in return value of `exp2(-1057)`, where
    // in VS 2013 it returns a double with the bit pattern 0x2 and in VS 2015 it
    // returns 0x20000.
    //
    // For now just ignore this test entirely on MSVC as it's tested elsewhere
    // anyway and we're not super interested in testing each platform's exp2
    // implementation.
    if !cfg!(target_env = "msvc") {
        f64_exact_sanity_test(format_exact);
    }
    f32_exact_sanity_test(format_exact);
}

#[test]
fn test_to_shortest_str() {
    to_shortest_str_test(format_shortest);
}

#[test]
fn test_to_shortest_exp_str() {
    to_shortest_exp_str_test(format_shortest);
}

#[test]
fn test_to_exact_exp_str() {
    to_exact_exp_str_test(format_exact);
}

#[test]
fn test_to_exact_fixed_str() {
    to_exact_fixed_str_test(format_exact);
}
