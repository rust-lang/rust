//! Decodes a floating-point value into individual parts and error ranges.

use crate::mem::size_of;

/// Generic decoding of floating points up to 64-bit wide such that its absolute
/// finite value matches mant * 2^exp. Values in range (mant - minus) * 2^exp up
/// to (mant + plus) * 2^exp will all round to the same value. The range with
/// minus and plus is inclusive only when `inclusive` is true.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Decoded64 {
    /// Scaled Mantissa
    pub mant: u64,
    /// Lower Error Range
    pub minus: u64,
    /// Upper Error Range
    pub plus: u64,
    /// Shared Exponent In Base 2
    pub exp: isize,
    /// Flag For Error Range
    pub inclusive: bool,
}

macro_rules! floats {
    ($($T:ident)*) => {
        $(

        /// Decode a floating-point into its integer components. The tuple in
        /// return contains the mantissa m and exponent e, such that original
        /// value equals m × 2^e, ignoring the sign.
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
                // Denormal numbers use a special exponent of 1 − bias instead
                // of −bias.
                mant_enc << 1
            };

            const _: () = assert!(ENC_BITS <= 64);
            (mant as u64, exp)
        }

        /// Parse a finite value into the generic structure.
        pub fn ${concat(decode_, $T)}(v: $T) -> Decoded64 {
            let (mant, exp) = ${concat(mant_and_exp_, $T)}(v);
            let is_even = (mant & 1) == 0;

            if v.is_subnormal() {
                // neighbors: (mant - 2, exp) -- (mant, exp) -- (mant + 2, exp)
                return Decoded64 { mant: mant, minus: 1, plus: 1, exp: exp, inclusive: is_even };
            }
            debug_assert!(v.is_normal());

            const MIN_POS_MANT: u64 = ${concat(mant_and_exp_, $T)}($T::MIN_POSITIVE).0;
            const MIN_NEG_MANT: u64 = ${concat(mant_and_exp_, $T)}(-$T::MIN_POSITIVE).0;
            const _: () = assert!(MIN_POS_MANT == MIN_NEG_MANT);
            if mant == MIN_POS_MANT {
                // neighbors: (maxmant, exp - 1) -- (minnormmant, exp) -- (minnormmant + 1, exp)
                // where maxmant = minnorm.mant * 2 - 1
                Decoded64 { mant: mant << 2, minus: 1, plus: 2, exp: exp - 2, inclusive: is_even }
            } else {
                // neighbors: (mant - 1, exp) -- (mant, exp) -- (mant + 1, exp)
                Decoded64 { mant: mant << 1, minus: 1, plus: 1, exp: exp - 1, inclusive: is_even }
            }
        }

        )*
    };
}

floats! { f16 f32 f64 }
