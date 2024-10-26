use core::ops;

use super::int_traits::{Int, MinInt};

/// Trait for some basic operations on floats
#[allow(dead_code)]
pub trait Float:
    Copy
    + core::fmt::Debug
    + PartialEq
    + PartialOrd
    + ops::AddAssign
    + ops::MulAssign
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Div<Output = Self>
    + ops::Rem<Output = Self>
{
    /// A uint of the same width as the float
    type Int: Int<OtherSign = Self::SignedInt, UnsignedInt = Self::Int>;

    /// A int of the same width as the float
    type SignedInt: Int + MinInt<OtherSign = Self::Int, UnsignedInt = Self::Int>;

    /// An int capable of containing the exponent bits plus a sign bit. This is signed.
    type ExpInt: Int;

    const ZERO: Self;
    const ONE: Self;

    /// The bitwidth of the float type
    const BITS: u32;

    /// The bitwidth of the significand
    const SIGNIFICAND_BITS: u32;

    /// The bitwidth of the exponent
    const EXPONENT_BITS: u32 = Self::BITS - Self::SIGNIFICAND_BITS - 1;

    /// The saturated value of the exponent (infinite representation), in the rightmost postiion.
    const EXPONENT_MAX: u32 = (1 << Self::EXPONENT_BITS) - 1;

    /// The exponent bias value
    const EXPONENT_BIAS: u32 = Self::EXPONENT_MAX >> 1;

    /// A mask for the sign bit
    const SIGN_MASK: Self::Int;

    /// A mask for the significand
    const SIGNIFICAND_MASK: Self::Int;

    /// The implicit bit of the float format
    const IMPLICIT_BIT: Self::Int;

    /// A mask for the exponent
    const EXPONENT_MASK: Self::Int;

    /// Returns `self` transmuted to `Self::Int`
    fn to_bits(self) -> Self::Int;

    /// Returns `self` transmuted to `Self::SignedInt`
    fn to_bits_signed(self) -> Self::SignedInt;

    /// Checks if two floats have the same bit representation. *Except* for NaNs! NaN can be
    /// represented in multiple different ways. This method returns `true` if two NaNs are
    /// compared.
    fn eq_repr(self, rhs: Self) -> bool;

    /// Returns true if the sign is negative
    fn is_sign_negative(self) -> bool;

    /// Returns the exponent, not adjusting for bias.
    fn exp(self) -> Self::ExpInt;

    /// Returns the significand with no implicit bit (or the "fractional" part)
    fn frac(self) -> Self::Int;

    /// Returns the significand with implicit bit
    fn imp_frac(self) -> Self::Int;

    /// Returns a `Self::Int` transmuted back to `Self`
    fn from_bits(a: Self::Int) -> Self;

    /// Constructs a `Self` from its parts. Inputs are treated as bits and shifted into position.
    fn from_parts(negative: bool, exponent: Self::Int, significand: Self::Int) -> Self;

    fn abs(self) -> Self {
        let abs_mask = !Self::SIGN_MASK;
        Self::from_bits(self.to_bits() & abs_mask)
    }

    /// Returns (normalized exponent, normalized significand)
    fn normalize(significand: Self::Int) -> (i32, Self::Int);

    /// Returns if `self` is subnormal
    fn is_subnormal(self) -> bool;
}

macro_rules! float_impl {
    ($ty:ident, $ity:ident, $sity:ident, $expty:ident, $bits:expr, $significand_bits:expr) => {
        impl Float for $ty {
            type Int = $ity;
            type SignedInt = $sity;
            type ExpInt = $expty;

            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;

            const BITS: u32 = $bits;
            const SIGNIFICAND_BITS: u32 = $significand_bits;

            const SIGN_MASK: Self::Int = 1 << (Self::BITS - 1);
            const SIGNIFICAND_MASK: Self::Int = (1 << Self::SIGNIFICAND_BITS) - 1;
            const IMPLICIT_BIT: Self::Int = 1 << Self::SIGNIFICAND_BITS;
            const EXPONENT_MASK: Self::Int = !(Self::SIGN_MASK | Self::SIGNIFICAND_MASK);

            fn to_bits(self) -> Self::Int {
                self.to_bits()
            }
            fn to_bits_signed(self) -> Self::SignedInt {
                self.to_bits() as Self::SignedInt
            }
            fn eq_repr(self, rhs: Self) -> bool {
                fn is_nan(x: $ty) -> bool {
                    // When using mangled-names, the "real" compiler-builtins might not have the
                    // necessary builtin (__unordtf2) to test whether `f128` is NaN.
                    // FIXME(f16_f128): Remove once the nightly toolchain has the __unordtf2 builtin
                    // x is NaN if all the bits of the exponent are set and the significand is non-0
                    x.to_bits() & $ty::EXPONENT_MASK == $ty::EXPONENT_MASK
                        && x.to_bits() & $ty::SIGNIFICAND_MASK != 0
                }
                if is_nan(self) && is_nan(rhs) { true } else { self.to_bits() == rhs.to_bits() }
            }
            fn is_sign_negative(self) -> bool {
                self.is_sign_negative()
            }
            fn exp(self) -> Self::ExpInt {
                ((self.to_bits() & Self::EXPONENT_MASK) >> Self::SIGNIFICAND_BITS) as Self::ExpInt
            }
            fn frac(self) -> Self::Int {
                self.to_bits() & Self::SIGNIFICAND_MASK
            }
            fn imp_frac(self) -> Self::Int {
                self.frac() | Self::IMPLICIT_BIT
            }
            fn from_bits(a: Self::Int) -> Self {
                Self::from_bits(a)
            }
            fn from_parts(negative: bool, exponent: Self::Int, significand: Self::Int) -> Self {
                Self::from_bits(
                    ((negative as Self::Int) << (Self::BITS - 1))
                        | ((exponent << Self::SIGNIFICAND_BITS) & Self::EXPONENT_MASK)
                        | (significand & Self::SIGNIFICAND_MASK),
                )
            }
            fn normalize(significand: Self::Int) -> (i32, Self::Int) {
                let shift = significand.leading_zeros().wrapping_sub(Self::EXPONENT_BITS);
                (1i32.wrapping_sub(shift as i32), significand << shift as Self::Int)
            }
            fn is_subnormal(self) -> bool {
                (self.to_bits() & Self::EXPONENT_MASK) == Self::Int::ZERO
            }
        }
    };
}

float_impl!(f32, u32, i32, i16, 32, 23);
float_impl!(f64, u64, i64, i16, 64, 52);
