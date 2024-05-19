#![allow(dead_code)] // FIXME(f16_f128): remove once constants are used

// We run out of precision pretty quickly with f16
const F16_APPROX_L1: f16 = 0.001;
const F16_APPROX_L2: f16 = 0.01;
const F16_APPROX_L3: f16 = 0.1;
const F16_APPROX_L4: f16 = 0.5;

/// Smallest number
const TINY_BITS: u16 = 0x1;
/// Next smallest number
const TINY_UP_BITS: u16 = 0x2;
/// Exponent = 0b11...10, Sifnificand 0b1111..10. Min val > 0
const MAX_DOWN_BITS: u16 = 0x7bfe;
/// Zeroed exponent, full significant
const LARGEST_SUBNORMAL_BITS: u16 = 0x03ff;
/// Exponent = 0b1, zeroed significand
const SMALLEST_NORMAL_BITS: u16 = 0x0400;
/// First pattern over the mantissa
const NAN_MASK1: u16 = 0x02aa;
/// Second pattern over the mantissa
const NAN_MASK2: u16 = 0x0155;

/// Compare by value
#[allow(unused_macros)]
macro_rules! assert_f16_eq {
    ($a:expr, $b:expr) => {
        let (l, r): (&f16, &f16) = (&$a, &$b);
        assert_eq!(*l, *r, "\na: {:#018x}\nb: {:#018x}", l.to_bits(), r.to_bits())
    };
}

/// Compare by representation
#[allow(unused_macros)]
macro_rules! assert_f16_biteq {
    ($a:expr, $b:expr) => {
        let (l, r): (&f16, &f16) = (&$a, &$b);
        let lb = l.to_bits();
        let rb = r.to_bits();
        assert_eq!(
            lb, rb,
            "float {:?} is not bitequal to {:?}.\na: {:#018x}\nb: {:#018x}",
            *l, *r, lb, rb
        );
    };
}
