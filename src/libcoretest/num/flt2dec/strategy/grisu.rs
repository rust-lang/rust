// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{i16, f64};
use super::super::*;
use core::num::flt2dec::strategy::grisu::*;

#[test]
fn test_cached_power() {
    assert_eq!(CACHED_POW10.first().unwrap().1, CACHED_POW10_FIRST_E);
    assert_eq!(CACHED_POW10.last().unwrap().1, CACHED_POW10_LAST_E);

    for e in -1137..961 { // full range for f64
        let low = ALPHA - e - 64;
        let high = GAMMA - e - 64;
        let (_k, cached) = cached_power(low, high);
        assert!(low <= cached.e && cached.e <= high,
                "cached_power({}, {}) = {:?} is incorrect", low, high, cached);
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


#[test]
fn shortest_sanity_test() {
    f64_shortest_sanity_test(format_shortest);
    f32_shortest_sanity_test(format_shortest);
    more_shortest_sanity_test(format_shortest);
}

#[test]
fn shortest_random_equivalence_test() {
    use core::num::flt2dec::strategy::dragon::format_shortest as fallback;
    f64_random_equivalence_test(format_shortest_opt, fallback, MAX_SIG_DIGITS, 10_000);
    f32_random_equivalence_test(format_shortest_opt, fallback, MAX_SIG_DIGITS, 10_000);
}

#[test] #[ignore] // it is too expensive
fn shortest_f32_exhaustive_equivalence_test() {
    // it is hard to directly test the optimality of the output, but we can at least test if
    // two different algorithms agree to each other.
    //
    // this reports the progress and the number of f32 values returned `None`.
    // with `--nocapture` (and plenty of time and appropriate rustc flags), this should print:
    // `done, ignored=17643158 passed=2121451881 failed=0`.

    use core::num::flt2dec::strategy::dragon::format_shortest as fallback;
    f32_exhaustive_equivalence_test(format_shortest_opt, fallback, MAX_SIG_DIGITS);
}

#[test] #[ignore] // it is too expensive
fn shortest_f64_hard_random_equivalence_test() {
    // this again probably has to use appropriate rustc flags.

    use core::num::flt2dec::strategy::dragon::format_shortest as fallback;
    f64_random_equivalence_test(format_shortest_opt, fallback,
                                         MAX_SIG_DIGITS, 100_000_000);
}

#[test]
fn exact_sanity_test() {
    // See comments in dragon.rs's exact_sanity_test for why this test is
    // ignored on MSVC
    if !cfg!(target_env = "msvc") {
        f64_exact_sanity_test(format_exact);
    }
    f32_exact_sanity_test(format_exact);
}

#[test]
fn exact_f32_random_equivalence_test() {
    use core::num::flt2dec::strategy::dragon::format_exact as fallback;
    for k in 1..21 {
        f32_random_equivalence_test(|d, buf| format_exact_opt(d, buf, i16::MIN),
                                             |d, buf| fallback(d, buf, i16::MIN), k, 1_000);
    }
}

#[test]
fn exact_f64_random_equivalence_test() {
    use core::num::flt2dec::strategy::dragon::format_exact as fallback;
    for k in 1..21 {
        f64_random_equivalence_test(|d, buf| format_exact_opt(d, buf, i16::MIN),
                                             |d, buf| fallback(d, buf, i16::MIN), k, 1_000);
    }
}

#[bench]
fn bench_small_shortest(b: &mut Bencher) {
    let decoded = decode_finite(3.141592f64);
    let mut buf = [0; MAX_SIG_DIGITS];
    b.iter(|| format_shortest(&decoded, &mut buf));
}

#[bench]
fn bench_big_shortest(b: &mut Bencher) {
    let decoded = decode_finite(f64::MAX);
    let mut buf = [0; MAX_SIG_DIGITS];
    b.iter(|| format_shortest(&decoded, &mut buf));
}

#[bench]
fn bench_small_exact_3(b: &mut Bencher) {
    let decoded = decode_finite(3.141592f64);
    let mut buf = [0; 3];
    b.iter(|| format_exact(&decoded, &mut buf, i16::MIN));
}

#[bench]
fn bench_big_exact_3(b: &mut Bencher) {
    let decoded = decode_finite(f64::MAX);
    let mut buf = [0; 3];
    b.iter(|| format_exact(&decoded, &mut buf, i16::MIN));
}

#[bench]
fn bench_small_exact_12(b: &mut Bencher) {
    let decoded = decode_finite(3.141592f64);
    let mut buf = [0; 12];
    b.iter(|| format_exact(&decoded, &mut buf, i16::MIN));
}

#[bench]
fn bench_big_exact_12(b: &mut Bencher) {
    let decoded = decode_finite(f64::MAX);
    let mut buf = [0; 12];
    b.iter(|| format_exact(&decoded, &mut buf, i16::MIN));
}

#[bench]
fn bench_small_exact_inf(b: &mut Bencher) {
    let decoded = decode_finite(3.141592f64);
    let mut buf = [0; 1024];
    b.iter(|| format_exact(&decoded, &mut buf, i16::MIN));
}

#[bench]
fn bench_big_exact_inf(b: &mut Bencher) {
    let decoded = decode_finite(f64::MAX);
    let mut buf = [0; 1024];
    b.iter(|| format_exact(&decoded, &mut buf, i16::MIN));
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

