//! Decodes a floating-point value into individual parts and error ranges.

use crate::num::FpCategory;
use crate::num::imp::FloatExt;

/// Decoded unsigned finite value, such that:
///
/// - The original value equals to `mant * 2^exp`.
///
/// - Any number from `(mant - minus) * 2^exp` to `(mant + plus) * 2^exp` will
///   round to the original value. The range is inclusive only when
///   `inclusive` is `true`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Decoded {
    /// The scaled mantissa.
    pub mant: u64,
    /// The lower error range.
    pub minus: u64,
    /// The upper error range.
    pub plus: u64,
    /// The shared exponent in base 2.
    pub exp: i16,
    /// True when the error range is inclusive.
    ///
    /// In IEEE 754, this is true when the original mantissa was even.
    pub inclusive: bool,
}

/// A floating point type which can be `decode`d.
pub trait DecodableFloat: FloatExt + Copy {
    /// The minimum positive normalized value.
    fn min_pos_norm_value() -> Self;
}

#[cfg(target_has_reliable_f16)]
impl DecodableFloat for f16 {
    fn min_pos_norm_value() -> Self {
        f16::MIN_POSITIVE
    }
}

impl DecodableFloat for f32 {
    fn min_pos_norm_value() -> Self {
        f32::MIN_POSITIVE
    }
}

impl DecodableFloat for f64 {
    fn min_pos_norm_value() -> Self {
        f64::MIN_POSITIVE
    }
}

/// Returns a `Decoded` value from given floating point number.
pub fn decode<T: DecodableFloat>(v: T) -> Decoded {
    let (mant, exp, _) = v.integer_decode();
    let even = (mant & 1) == 0;
    match v.classify() {
        FpCategory::Subnormal => {
            // neighbors: (mant - 2, exp) -- (mant, exp) -- (mant + 2, exp)
            // Float::integer_decode always preserves the exponent,
            // so the mantissa is scaled for subnormals.
            Decoded { mant, minus: 1, plus: 1, exp, inclusive: even }
        }
        FpCategory::Normal => {
            let minnorm = <T as DecodableFloat>::min_pos_norm_value().integer_decode();
            if mant == minnorm.0 {
                // neighbors: (maxmant, exp - 1) -- (minnormmant, exp) -- (minnormmant + 1, exp)
                // where maxmant = minnormmant * 2 - 1
                Decoded { mant: mant << 2, minus: 1, plus: 2, exp: exp - 2, inclusive: even }
            } else {
                // neighbors: (mant - 1, exp) -- (mant, exp) -- (mant + 1, exp)
                Decoded { mant: mant << 1, minus: 1, plus: 1, exp: exp - 1, inclusive: even }
            }
        }
        _ => panic!("need a finite non-zero value"),
    }
}
