//! Slow, fallback algorithm for cases the Eisel-Lemire algorithm cannot round.

use crate::num::dec2flt::common::BiasedFp;
use crate::num::dec2flt::decimal_seq::{DecimalSeq, parse_decimal_seq};
use crate::num::dec2flt::float::RawFloat;

/// Parse the significant digits and biased, binary exponent of a float.
///
/// This is a fallback algorithm that uses a big-integer representation
/// of the float, and therefore is considerably slower than faster
/// approximations. However, it will always determine how to round
/// the significant digits to the nearest machine float, allowing
/// use to handle near half-way cases.
///
/// Near half-way cases are halfway between two consecutive machine floats.
/// For example, the float `16777217.0` has a bitwise representation of
/// `100000000000000000000000 1`. Rounding to a single-precision float,
/// the trailing `1` is truncated. Using round-nearest, tie-even, any
/// value above `16777217.0` must be rounded up to `16777218.0`, while
/// any value before or equal to `16777217.0` must be rounded down
/// to `16777216.0`. These near-halfway conversions therefore may require
/// a large number of digits to unambiguously determine how to round.
///
/// The algorithms described here are based on "Processing Long Numbers Quickly",
/// available here: <https://arxiv.org/pdf/2101.11408.pdf#section.11>.
pub(crate) fn parse_long_mantissa<F: RawFloat>(s: &[u8]) -> BiasedFp {
    const MAX_SHIFT: usize = 60;
    const NUM_POWERS: usize = 19;
    const POWERS: [u8; 19] =
        [0, 3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 43, 46, 49, 53, 56, 59];

    let get_shift = |n| {
        if n < NUM_POWERS { POWERS[n] as usize } else { MAX_SHIFT }
    };

    let fp_zero = BiasedFp::zero_pow2(0);
    let fp_inf = BiasedFp::zero_pow2(F::INFINITE_POWER);

    let mut d = parse_decimal_seq(s);

    // Short-circuit if the value can only be a literal 0 or infinity.
    if d.num_digits == 0 || d.decimal_point < -324 {
        return fp_zero;
    } else if d.decimal_point >= 310 {
        return fp_inf;
    }
    let mut exp2 = 0_i32;
    // Shift right toward (1/2 ... 1].
    while d.decimal_point > 0 {
        let n = d.decimal_point as usize;
        let shift = get_shift(n);
        d.right_shift(shift);
        if d.decimal_point < -DecimalSeq::DECIMAL_POINT_RANGE {
            return fp_zero;
        }
        exp2 += shift as i32;
    }
    // Shift left toward (1/2 ... 1].
    while d.decimal_point <= 0 {
        let shift = if d.decimal_point == 0 {
            match d.digits[0] {
                digit if digit >= 5 => break,
                0 | 1 => 2,
                _ => 1,
            }
        } else {
            get_shift((-d.decimal_point) as _)
        };
        d.left_shift(shift);
        if d.decimal_point > DecimalSeq::DECIMAL_POINT_RANGE {
            return fp_inf;
        }
        exp2 -= shift as i32;
    }
    // We are now in the range [1/2 ... 1] but the binary format uses [1 ... 2].
    exp2 -= 1;
    while F::EXP_MIN > exp2 {
        let mut n = (F::EXP_MIN - exp2) as usize;
        if n > MAX_SHIFT {
            n = MAX_SHIFT;
        }
        d.right_shift(n);
        exp2 += n as i32;
    }
    if (exp2 - F::EXP_MIN + 1) >= F::INFINITE_POWER {
        return fp_inf;
    }
    // Shift the decimal to the hidden bit, and then round the value
    // to get the high mantissa+1 bits.
    d.left_shift(F::SIG_BITS as usize + 1);
    let mut mantissa = d.round();
    if mantissa >= (1_u64 << (F::SIG_BITS + 1)) {
        // Rounding up overflowed to the carry bit, need to
        // shift back to the hidden bit.
        d.right_shift(1);
        exp2 += 1;
        mantissa = d.round();
        if (exp2 - F::EXP_MIN + 1) >= F::INFINITE_POWER {
            return fp_inf;
        }
    }
    let mut power2 = exp2 - F::EXP_MIN + 1;
    if mantissa < (1_u64 << F::SIG_BITS) {
        power2 -= 1;
    }
    // Zero out all the bits above the explicit mantissa bits.
    mantissa &= (1_u64 << F::SIG_BITS) - 1;
    BiasedFp { m: mantissa, p_biased: power2 }
}
