#![allow(dead_code)] // FIXME(f16_f128): remove once constants are used

/// Smallest number
const TINY_BITS: u128 = 0x1;
/// Next smallest number
const TINY_UP_BITS: u128 = 0x2;
/// Exponent = 0b11...10, Sifnificand 0b1111..10. Min val > 0
const MAX_DOWN_BITS: u128 = 0x7ffeffffffffffffffffffffffffffff;
/// Zeroed exponent, full significant
const LARGEST_SUBNORMAL_BITS: u128 = 0x0000ffffffffffffffffffffffffffff;
/// Exponent = 0b1, zeroed significand
const SMALLEST_NORMAL_BITS: u128 = 0x00010000000000000000000000000000;
/// First pattern over the mantissa
const NAN_MASK1: u128 = 0x0000aaaaaaaaaaaaaaaaaaaaaaaaaaaa;
/// Second pattern over the mantissa
const NAN_MASK2: u128 = 0x00005555555555555555555555555555;

/// Compare by value
#[allow(unused_macros)]
macro_rules! assert_f128_eq {
    ($a:expr, $b:expr) => {
        let (l, r): (&f128, &f128) = (&$a, &$b);
        assert_eq!(*l, *r, "\na: {:#0130x}\nb: {:#0130x}", l.to_bits(), r.to_bits())
    };
}

/// Compare by representation
#[allow(unused_macros)]
macro_rules! assert_f128_biteq {
    ($a:expr, $b:expr) => {
        let (l, r): (&f128, &f128) = (&$a, &$b);
        let lb = l.to_bits();
        let rb = r.to_bits();
        assert_eq!(
            lb, rb,
            "float {:?} is not bitequal to {:?}.\na: {:#0130x}\nb: {:#0130x}",
            *l, *r, lb, rb
        );
    };
}
