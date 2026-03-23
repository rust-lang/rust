use core::num::imp::flt2dec;
use core::num::imp::flt2dec::strategy::grisu;

use super::super::*;

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn test_cached_power() {
    assert_eq!(grisu::CACHED_POW10.first().unwrap().1, grisu::CACHED_POW10_FIRST_E);
    assert_eq!(grisu::CACHED_POW10.last().unwrap().1, grisu::CACHED_POW10_LAST_E);

    for e in -1137..961 {
        // full range for f64
        let low = grisu::ALPHA - e - 64;
        let high = grisu::GAMMA - e - 64;
        let (_k, cached) = grisu::cached_power(low, high);
        assert!(
            low <= cached.e && cached.e <= high,
            "cached_power({}, {}) = {:?} is incorrect",
            low,
            high,
            cached
        );
    }
}

#[test]
fn test_max_pow10_no_more_than() {
    let mut prevtenk = 1;
    for k in 1..10 {
        let tenk = prevtenk * 10;
        assert_eq!(grisu::max_pow10_no_more_than(tenk - 1), (k - 1, prevtenk));
        assert_eq!(grisu::max_pow10_no_more_than(tenk), (k, tenk));
        prevtenk = tenk;
    }
}

#[test]
fn shortest_sanity_test() {
    f64_shortest_sanity_test(flt2dec::format_short);
    f32_shortest_sanity_test(flt2dec::format_short);
    #[cfg(target_has_reliable_f16)]
    f16_shortest_sanity_test(flt2dec::format_short);
    more_shortest_sanity_test(flt2dec::format_short);
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn exact_sanity_test() {
    // See comments in dragon.rs's exact_sanity_test for why this test is
    // ignored on MSVC
    if !cfg!(target_env = "msvc") {
        f64_exact_sanity_test(flt2dec::format_fixed);
    }
    f32_exact_sanity_test(flt2dec::format_fixed);
    #[cfg(target_has_reliable_f16)]
    f16_exact_sanity_test(flt2dec::format_fixed);
}

#[test]
fn test_to_shortest_str() {
    to_shortest_str_test(flt2dec::format_short);
}

#[test]
fn test_to_shortest_exp_str() {
    to_shortest_exp_str_test(flt2dec::format_short);
}

#[test]
fn test_to_exact_exp_str() {
    to_exact_exp_str_test(flt2dec::format_fixed);
}

#[test]
fn test_to_exact_fixed_str() {
    to_exact_fixed_str_test(flt2dec::format_fixed);
}
