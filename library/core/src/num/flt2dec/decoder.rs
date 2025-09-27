//! Decodes a floating-point value into individual parts and error ranges.

use crate::mem::size_of;

/// Generic representation of finite floating-point values up to 64-bit wide.
/// The absolute value equals `mant * 2^exp`. All real values `x` such that:
///
///  lower < x < upper
///
/// (or `lower < x ≤ upper` in the tie-to-even case) round to this value under
/// IEEE 754 round-to-nearest rules, where:
///
///  lower = (mant − minus) * 2^exp
///  upper = (mant + plus)  * 2^exp
///
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Decoded64 {
    /// Scaled mantissa. The scaling is chosen such that the rounding boundaries
    /// are integral when expressed as `mant ± {minus, plus}`.
    pub mant: u64,

    /// Distance from `mant` to the lower rounding-boundary.
    pub minus: u64,
    /// Distance from `mant` to the upper rounding-boundary.
    pub plus: u64,

    /// Base-2 exponent for `mant`, `minus`, and `plus`.
    pub exp: isize,

    /// Indicates whether an exact tie at the upper rounding-boundary (i.e., the
    /// midpoint between this value and its next representable neighbor) rounds
    /// to this value. It applies only to the upper boundary; the lower boundary
    /// is always exclusive. This follows IEEE 754 round-ties-to-even semantics
    /// and is true iff the original significand was even.
    pub tie_to_even: bool,
}

macro_rules! floats {
    ($($T:ident)*) => {
        $(

        /// Decode a floating-point into its integer components. The tuple in
        /// return contains the mantissa m and exponent e, such that original
        /// value equals `m * 2^e`, ignoring the sign.
        ///
        /// For normal numbers: mantissa includes the implied leading 1.
        /// For denormal numbers: mantissa is shifted to maintain the equation.
        const fn ${concat(mant_and_exp_, $T)}(v: $T) -> (u64, isize) {
            const ENC_BITS: usize = size_of::<$T>() * 8;
            // The encoding of the sign resides in the most significant bit.
            const SIGN_ENC_BITS: usize = 1;
            // The encoding of the mantissa resides in the least-significant
            // bits.
            const MANT_ENC_BITS: usize = $T::MANTISSA_DIGITS as usize - 1;
            // The encoding of the exponent resides in the remaining bits,
            // inbetween sign and the mantissa.
            const EXP_ENC_BITS: usize = ENC_BITS - (SIGN_ENC_BITS + MANT_ENC_BITS);

            let enc = v.to_bits();
            let exp_enc = (enc << SIGN_ENC_BITS) >> (SIGN_ENC_BITS + MANT_ENC_BITS);
            let mant_enc = enc & ((1 << MANT_ENC_BITS) - 1);

            const EXP_BIAS: isize = (1 << (EXP_ENC_BITS - 1)) - 1;
            let exp = exp_enc as isize - (EXP_BIAS + MANT_ENC_BITS as isize);

            let mant = if exp_enc != 0 {
                // Normal numbers have an implied leading 1 to the mantissa
                // bits.
                mant_enc | 1 << MANT_ENC_BITS
            } else {
                // Account for the effective +1 on exponents of denormal numbers.
                mant_enc << 1
            };

            const _: () = assert!(ENC_BITS <= 64);
            (mant as u64, exp)
        }

        /// Parse a finite, non-zero floating-point into the generic structure.
        pub fn ${concat(decode_, $T)}(v: $T) -> Decoded64 {
            let (mant, exp) = ${concat(mant_and_exp_, $T)}(v);

            if v.is_subnormal() {
                // Subnormal floats have doubled spacing between representable values.
                // The boundaries are symmetric around `mant` in this scaled integer space.
                return Decoded64 { mant: mant, minus: 1, plus: 1, exp: exp, tie_to_even: true };
            }
            debug_assert!(v.is_normal());

            let is_even = (mant & 1) == 0;

            const MIN_POS_MANT: u64 = ${concat(mant_and_exp_, $T)}($T::MIN_POSITIVE).0;
            const _: () = assert!(MIN_POS_MANT != 0);
            if mant == MIN_POS_MANT {
                // The previous float of the first normal number is the largest subnormal.
                // The upper boundary is asymmetrically farther away than the lower boundary.
                Decoded64 { mant: mant << 2, minus: 1, plus: 2, exp: exp - 2, tie_to_even: is_even }
            } else {
                Decoded64 { mant: mant << 1, minus: 1, plus: 1, exp: exp - 1, tie_to_even: is_even }
            }
        }

        )*
    };
}

floats! { f16 f32 f64 }
