//! Software floating point functions for `f16`.

#[cfg(test)]
mod tests;

/// The floating point type.
type FP = f16;

/// Type of the exponent.
type Exponent = i16;

/// Type of `FP::to_bits`.
type Bits = u16;

/// Half as many bits as `Bits`.
type HalfBits = u8;

/// Twice as many bits as `Bits`.
type DoubleBits = u32;

/// Number of bits in the significand.
const SIGNIFICAND_BITS: u32 = FP::MANTISSA_DIGITS - 1;

/// Number of bits in the exponent.
const EXPONENT_BITS: u32 = Bits::BITS - 1 - SIGNIFICAND_BITS;

/// Encoded exponent = EXPONENT_BIAS + actual exponent.
const EXPONENT_BIAS: Exponent = 2 - FP::MIN_EXP as Exponent;

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
        // We have to use `as` because `From<u8> for f16` is not yet implemented.
        q as f16
    } else if exp <= FP::MAX_EXP as Exponent {
        // Verify that `aa` fits in `DoubleBits`.
        const _: () = assert!(FP::MAX_EXP as u32 <= Bits::BITS);
        let aa = DoubleBits::from(a.mantissa) << exp;
        let bb = DoubleBits::from(b.mantissa);
        let q = aa / bb;
        q as FP
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
        // We have to use `as` because `From<u8> for f16` is not yet implemented.
        q as f16
    } else if exp <= FP::MAX_EXP as Exponent {
        // Verify that `aa` fits in `DoubleBits`.
        const _: () = assert!(FP::MAX_EXP as u32 <= Bits::BITS);
        let aa = DoubleBits::from(a.mantissa) << exp;
        let bb = DoubleBits::from(b.mantissa);
        let q = (aa - 1) / bb + 1;
        q as FP
    } else {
        // a.mantissa / b.mantissa >= 1/2, hence
        // a / b >= 1/2 * 2^(MAX_EXP+1) = 2^MAX_EXP
        FP::INFINITY
    }
}
