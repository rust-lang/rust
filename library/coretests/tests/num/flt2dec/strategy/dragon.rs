use core::num::imp::bignum::Big32x40 as Big;
use core::num::imp::flt2dec::strategy::dragon::*;

use super::super::*;

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

#[test]
fn test_to_shortest_str() {
    to_shortest_str_test(format_short);
}

#[test]
fn test_to_shortest_exp_str() {
    to_shortest_exp_str_test(format_short);
}

#[test]
fn test_to_exact_exp_str() {
    to_exact_exp_str_test(format_fixed);
}

#[test]
fn test_to_exact_fixed_str() {
    to_exact_fixed_str_test(format_fixed);
}
