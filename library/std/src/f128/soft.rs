//! Software floating point functions for `f128`.

#[cfg(test)]
mod tests;

use crate::u256::U256;

/// The floating point type.
type FP = f128;

/// Type of the exponent.
type Exponent = i32;

/// Type of `FP::to_bits`.
type Bits = u128;

/// Half as many bits as `Bits`.
type HalfBits = u64;

/// Twice as many bits as `Bits`.
type DoubleBits = U256;

/// Number of bits in the significand.
const SIGNIFICAND_BITS: u32 = FP::MANTISSA_DIGITS - 1;

/// Number of bits in the exponent.
const EXPONENT_BITS: u32 = Bits::BITS - 1 - SIGNIFICAND_BITS;

/// Encoded exponent = EXPONENT_BIAS + actual exponent.
const EXPONENT_BIAS: Exponent = 2 - FP::MIN_EXP;

/// Represents an `FP` number.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Representation {
    /// Sign.
    sign: Sign,
    /// Absolute value.
    abs: Positive,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Sign {
    Positive,
    Negative,
}

/// Represents a positive number.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Positive {
    Zero,
    Finite(PositiveFinite),
    Infinity,
    NaN,
}

/// Represents a non-zero, positive finite number.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct PositiveFinite {
    /// `number = 2^exp * mantissa`.
    exp: Exponent,
    /// `2^SIGNIFICAND_BITS <= mantissa < 2^(SIGNIFICAND_BITS + 1)`
    mantissa: Bits,
}

impl Representation {
    fn new(x: FP) -> Self {
        let bits = x.to_bits();

        let sign = if (bits >> (SIGNIFICAND_BITS + EXPONENT_BITS)) & 1 == 0 {
            Sign::Positive
        } else {
            Sign::Negative
        };

        const SIGNIFICAND_MASK: Bits = (1 << SIGNIFICAND_BITS) - 1;
        const EXPONENT_MASK: Bits = (1 << EXPONENT_BITS) - 1;

        let significand = bits & SIGNIFICAND_MASK;
        let biased_exponent = bits >> SIGNIFICAND_BITS & EXPONENT_MASK;

        let abs = match biased_exponent {
            0 if significand == 0 => Positive::Zero,
            0 => {
                // Subnormal number.
                // Normalize it by shifting left, so that it has exactly
                // SIGNIFICAND_BITS + 1 bits.
                let shift = SIGNIFICAND_BITS + 1 - (Bits::BITS - significand.leading_zeros());

                Positive::Finite(PositiveFinite {
                    exp: 1 - EXPONENT_BIAS - SIGNIFICAND_BITS as Exponent - shift as Exponent,
                    mantissa: significand << shift,
                })
            }
            EXPONENT_MASK if significand == 0 => Positive::Infinity,
            EXPONENT_MASK => Positive::NaN,
            _ => Positive::Finite(PositiveFinite {
                exp: biased_exponent as Exponent - EXPONENT_BIAS - SIGNIFICAND_BITS as Exponent,
                mantissa: 1 << SIGNIFICAND_BITS | significand,
            }),
        };
        Representation { sign, abs }
    }
}

/// Shift right that works even if `shift >= Bits::BITS`.
///
/// Requires `shift >= 0`.
fn safe_shr(a: Bits, shift: Exponent) -> Bits {
    if shift < Bits::BITS as Exponent { a >> shift } else { 0 }
}

/// Returns `2^exp`.
fn pow2(exp: Exponent) -> FP {
    let biased_exponent = exp + EXPONENT_BIAS;
    if biased_exponent <= -(SIGNIFICAND_BITS as Exponent) {
        // Round to 0.
        0.0
    } else if biased_exponent <= 0 {
        // Subnormal.
        FP::from_bits(1 << (biased_exponent + (SIGNIFICAND_BITS as Exponent) - 1))
    } else if biased_exponent <= (1 << EXPONENT_BITS) - 1 {
        // Normal.
        FP::from_bits((biased_exponent as Bits) << SIGNIFICAND_BITS)
    } else {
        // Round to infinity.
        FP::INFINITY
    }
}

/// Euclidean division.
#[allow(dead_code)]
pub(crate) fn div_euclid(a: FP, b: FP) -> FP {
    let a = Representation::new(a);
    let b = Representation::new(b);
    let res = match a.sign {
        Sign::Positive => div_floor_pos(a.abs, b.abs),
        Sign::Negative => -div_ceil_pos(a.abs, b.abs),
    };
    match b.sign {
        Sign::Positive => res,
        Sign::Negative => -res,
    }
}

/// Division rounded down for positive numbers.
fn div_floor_pos(a: Positive, b: Positive) -> FP {
    match (a, b) {
        (Positive::NaN, _) => FP::NAN,
        (_, Positive::NaN) => FP::NAN,
        (Positive::Zero, Positive::Zero) => FP::NAN,
        (Positive::Zero, Positive::Finite(_)) => 0.0,
        (Positive::Zero, Positive::Infinity) => 0.0,
        (Positive::Finite(_), Positive::Zero) => FP::INFINITY,
        (Positive::Finite(a), Positive::Finite(b)) => div_floor_pos_finite(a, b),
        (Positive::Finite(_), Positive::Infinity) => 0.0,
        (Positive::Infinity, Positive::Zero) => FP::INFINITY,
        (Positive::Infinity, Positive::Finite(_)) => FP::INFINITY,
        (Positive::Infinity, Positive::Infinity) => FP::NAN,
    }
}

/// Division rounded up for positive numbers.
fn div_ceil_pos(a: Positive, b: Positive) -> FP {
    match (a, b) {
        (Positive::NaN, _) => FP::NAN,
        (_, Positive::NaN) => FP::NAN,
        (Positive::Zero, Positive::Zero) => FP::NAN,
        (Positive::Zero, Positive::Finite(_)) => 0.0,
        (Positive::Zero, Positive::Infinity) => 0.0,
        (Positive::Finite(_), Positive::Zero) => FP::INFINITY,
        (Positive::Finite(a), Positive::Finite(b)) => div_ceil_pos_finite(a, b),
        // Tricky case. It's 1.0 rather than 0.0. This way
        // `div_euclid(-finite, inf) = -1.0` is consistent with
        // `rem_euclid(-finite, inf) = inf`.
        (Positive::Finite(_), Positive::Infinity) => 1.0,
        (Positive::Infinity, Positive::Zero) => FP::INFINITY,
        (Positive::Infinity, Positive::Finite(_)) => FP::INFINITY,
        (Positive::Infinity, Positive::Infinity) => FP::NAN,
    }
}

/// Division rounded down for positive finite numbers.
fn div_floor_pos_finite(a: PositiveFinite, b: PositiveFinite) -> FP {
    let exp = a.exp - b.exp;
    if exp < 0 {
        0.0
    } else if exp <= (Bits::BITS - SIGNIFICAND_BITS - 1) as Exponent {
        // `aa` fits in `Bits`
        let aa = a.mantissa << exp;
        // a.mantissa / b.mantissa < 2, hence `q` fits in `HalfBits` as long as:
        // exp + 1 <= HalfBits::BITS
        // Bits::BITS - SIGNIFICAND_BITS <= HalfBits::BITS
        // SIGNIFICAND_BITS >= HalfBits::BITS
        const _: () = assert!(SIGNIFICAND_BITS >= HalfBits::BITS);
        let q = (aa / b.mantissa) as HalfBits;
        // We have to use `as` because `From<u64> for f128` is not yet implemented.
        q as f128
    } else if exp <= (Bits::BITS - 1) as Exponent {
        let aa = DoubleBits::from(a.mantissa) << exp as u32;
        let bb = DoubleBits::from(b.mantissa);
        // `q` fits in `Bits` because `a.mantissa < 2 * b.mantissa`.
        let q: Bits = (aa / bb).wrap_u128();
        q as FP
    } else if exp <= FP::MAX_EXP {
        // exp >= Bits::BITS
        let aa = DoubleBits::from(a.mantissa) << (Bits::BITS - 1);
        let bb = DoubleBits::from(b.mantissa);

        // e > 0
        // The result is `floor((aa << e) / bb)`
        let e = exp - (Bits::BITS - 1) as Exponent;

        // aa = q * bb + r
        // `q` fits in `Bits` because `a.mantissa < 2 * b.mantissa`.
        // `q > Bits::MAX / 4` because `b.mantissa < 2 * a.mantissa`.
        // 0 <= r < b
        let (q, r) = aa.div_rem(bb);
        let q: Bits = q.wrap_u128();
        let r: Bits = r.wrap_u128();

        // result = floor((aa << e) / bb) = (q << e) + floor((r << e) / bb)
        // 0 <= (r << e) / bb < 2^e
        //
        // There are two cases:
        // 1. floor((r << e) / bb) = 0
        // 2. 0 < floor((r << e) / bb) < 2^e
        //
        // Case 1:
        // The result is q << e.
        //
        // Case 2:
        // The result is (q << e) + non-zero low e bits.
        //
        // Rounding beyond the SIGNIFICAND_BITS + 2 most significant bits of q depends
        // only on whether the low-order bits are non-zero. Since q > Bits::MAX / 4,
        // q.leading_zeros() <= 1. Therefore, beyond the top SIGNIFICAND_BITS + 3 bits
        // of q, it doesn't matter *which* bits are non-zero. As long as:
        //
        // SIGNIFICAND_BITS <= Bits::BITS - 4
        //
        // we can just set bit 0 of q to 1 instead of using extra low-order bits.
        //
        // Therefore the result rounds the same way as (q | 1) << e.
        //
        // Case 1 happens when:
        // (r << e) / bb < 1
        // (r << e) <= bb - 1
        // r <= ((bb - 1) >> e)
        let case_1_bound = safe_shr(b.mantissa - 1, e);
        let q_adj = if r <= case_1_bound {
            // Case 1.
            q
        } else {
            // Case 2.
            const _: () = assert!(SIGNIFICAND_BITS + 4 <= Bits::BITS);
            q | 1
        };
        q_adj as FP * pow2(e)
    } else {
        // a.mantissa / b.mantissa >= 1/2, hence
        // a / b >= 1/2 * 2^(MAX_EXP+1) = 2^MAX_EXP
        FP::INFINITY
    }
}

/// Division rounded up for positive finite numbers.
fn div_ceil_pos_finite(a: PositiveFinite, b: PositiveFinite) -> FP {
    let exp = a.exp - b.exp;
    if exp < 0 {
        1.0
    } else if exp <= (Bits::BITS - SIGNIFICAND_BITS - 1) as Exponent {
        // `aa` fits in `Bits`
        let aa = a.mantissa << exp;
        // a.mantissa / b.mantissa < 2, hence `q` fits in `HalfBits` as long as:
        // exp + 1 < HalfBits::BITS
        // Bits::BITS - SIGNIFICAND_BITS < HalfBits::BITS
        // SIGNIFICAND_BITS > HalfBits::BITS
        const _: () = assert!(SIGNIFICAND_BITS > HalfBits::BITS);
        let q = ((aa - 1) / b.mantissa) as HalfBits + 1;
        // We have to use `as` because `From<u64> for f128` is not yet implemented.
        q as f128
    } else if exp <= (Bits::BITS - 1) as Exponent {
        let aa = DoubleBits::from(a.mantissa) << exp as u32;
        let bb = DoubleBits::from(b.mantissa);
        // q <= aa / bb + 1 <= (2 - 2^-SIGNIFICAND_BITS) * 2^(Bits::BITS-1) + 1
        //   <= 2^Bits::BITS - 2^(Bits::BITS - SIGNIFICAND_BITS) + 1 <= Bits::MAX
        let q = ((aa - U256::ONE) / bb).wrap_u128() + 1;
        q as FP
    } else if exp <= FP::MAX_EXP {
        let aa = DoubleBits::from(a.mantissa) << (Bits::BITS - 1);
        let bb = DoubleBits::from(b.mantissa);
        // e > 0
        // The result is ceil((aa << e) / b).
        let e = exp - (Bits::BITS - 1) as Exponent;

        // aa = q * bb + r
        // `q < Bits::MAX` as in the previous case.
        // `q > Bits::MAX / 4` because `b.mantissa < 2 * a.mantissa`.
        // 0 <= r < b
        let (q, r) = aa.div_rem(bb);
        let q: Bits = q.wrap_u128();
        let r: Bits = r.wrap_u128();

        // result = ceil((aa << e) / bb) = (q << e) + ceil((r << e) / bb)
        // 0 <= (r << e) / bb < 2^e
        //
        // There are three cases:
        // 1. ceil((r << e) / bb) = 0
        // 2. 0 < ceil((r << e) / bb) < 2^e
        // 3. ceil((r << e) / bb) = 2^e
        //
        // Case 1:
        // The result is q << e.
        //
        // Case 2:
        // The result is (q << e) + non-zero low e bits.
        //
        // Rounding beyond the SIGNIFICAND_BITS + 2 most significant bits of q depends
        // only on whether the low-order bits are non-zero. Since q > Bits::MAX / 4,
        // q.leading_zeros() <= 1. Therefore, beyond the top SIGNIFICAND_BITS + 3 bits
        // of q, it doesn't matter *which* bits are non-zero. As long as:
        //
        // SIGNIFICAND_BITS <= Bits::BITS - 4
        //
        // we can just set bit 0 of q to 1 instead of using extra low-order bits.
        //
        // Therefore the result rounds the same way as (q | 1) << e.
        //
        // In case 3:
        // The result is (q + 1) << e.
        //
        // Case 1 happens when r = 0.
        // Case 3 happens when:
        // (r << e) / bb > (1 << e) - 1
        // (r << e) > (bb << e) - b
        // ((bb - r) << e) <= bb - 1
        // bb - r <= (bb - 1) >> e
        // r >= bb - ((bb - 1) >> e)
        let case_3_bound = b.mantissa - safe_shr(b.mantissa - 1, e);
        let q_adj = if r == 0 {
            // Case 1.
            q
        } else if r < case_3_bound {
            // Case 2.
            const _: () = assert!(SIGNIFICAND_BITS + 4 <= Bits::BITS);
            q | 1
        } else {
            // Case 3.
            // No overflow because `q < Bits::MAX`.
            q + 1
        };
        q_adj as FP * pow2(e)
    } else {
        // a.mantissa / b.mantissa >= 1/2, hence
        // a / b >= 1/2 * 2^(MAX_EXP+1) = 2^MAX_EXP
        FP::INFINITY
    }
}
