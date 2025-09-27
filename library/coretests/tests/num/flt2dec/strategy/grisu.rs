use core::num::flt2dec::strategy::grisu::*;

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn test_cached_power() {
    assert_eq!(CACHED_POW10.first().unwrap().1, CACHED_POW10_FIRST_E);
    assert_eq!(CACHED_POW10.last().unwrap().1, CACHED_POW10_LAST_E);

    for e in -1137..961 {
        // full range for f64
        let low = ALPHA - e - 64;
        let high = GAMMA - e - 64;
        let (_k, cached) = cached_power(low, high);
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
        assert_eq!(max_pow10_no_more_than(tenk - 1), (k - 1, prevtenk));
        assert_eq!(max_pow10_no_more_than(tenk), (k, tenk));
        prevtenk = tenk;
    }
}
