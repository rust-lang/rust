use core::mem;
use core::ops;

use super::int::Int;

pub mod conv;
pub mod cmp;
pub mod add;
pub mod pow;
pub mod sub;
pub mod mul;
pub mod div;
pub mod extend;

/// Trait for some basic operations on floats
pub trait Float:
    Copy +
    PartialEq +
    PartialOrd +
    ops::AddAssign +
    ops::MulAssign +
    ops::Add<Output = Self> +
    ops::Sub<Output = Self> +
    ops::Div<Output = Self> +
    ops::Rem<Output = Self> +
{
    /// A uint of the same with as the float
    type Int: Int;

    /// A int of the same with as the float
    type SignedInt: Int;

    const ZERO: Self;
    const ONE: Self;

    /// The bitwidth of the float type
    const BITS: u32;

    /// The bitwidth of the significand
    const SIGNIFICAND_BITS: u32;

    /// The bitwidth of the exponent
    const EXPONENT_BITS: u32 = Self::BITS - Self::SIGNIFICAND_BITS - 1;

    /// The maximum value of the exponent
    const EXPONENT_MAX: u32 = (1 << Self::EXPONENT_BITS) - 1;

    /// The exponent bias value
    const EXPONENT_BIAS: u32 = Self::EXPONENT_MAX >> 1;

    /// A mask for the sign bit
    const SIGN_MASK: Self::Int;

    /// A mask for the significand
    const SIGNIFICAND_MASK: Self::Int;

    // The implicit bit of the float format
    const IMPLICIT_BIT: Self::Int;

    /// A mask for the exponent
    const EXPONENT_MASK: Self::Int;

    /// Returns `self` transmuted to `Self::Int`
    fn repr(self) -> Self::Int;

    /// Returns `self` transmuted to `Self::SignedInt`
    fn signed_repr(self) -> Self::SignedInt;

    #[cfg(test)]
    /// Checks if two floats have the same bit representation. *Except* for NaNs! NaN can be
    /// represented in multiple different ways. This method returns `true` if two NaNs are
    /// compared.
    fn eq_repr(self, rhs: Self) -> bool;

    /// Returns a `Self::Int` transmuted back to `Self`
    fn from_repr(a: Self::Int) -> Self;

    /// Constructs a `Self` from its parts. Inputs are treated as bits and shifted into position.
    fn from_parts(sign: bool, exponent: Self::Int, significand: Self::Int) -> Self;

    /// Returns (normalized exponent, normalized significand)
    fn normalize(significand: Self::Int) -> (i32, Self::Int);
}

// FIXME: Some of this can be removed if RFC Issue #1424 is resolved
//        https://github.com/rust-lang/rfcs/issues/1424
macro_rules! float_impl {
    ($ty:ident, $ity:ident, $sity:ident, $bits:expr, $significand_bits:expr) => {
        impl Float for $ty {
            type Int = $ity;
            type SignedInt = $sity;
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;

            const BITS: u32 = $bits;
            const SIGNIFICAND_BITS: u32 = $significand_bits;

            const SIGN_MASK: Self::Int = 1 << (Self::BITS - 1);
            const SIGNIFICAND_MASK: Self::Int = (1 << Self::SIGNIFICAND_BITS) - 1;
            const IMPLICIT_BIT: Self::Int = 1 << Self::SIGNIFICAND_BITS;
            const EXPONENT_MASK: Self::Int = !(Self::SIGN_MASK | Self::SIGNIFICAND_MASK);

            fn repr(self) -> Self::Int {
                unsafe { mem::transmute(self) }
            }
            fn signed_repr(self) -> Self::SignedInt {
                unsafe { mem::transmute(self) }
            }
            #[cfg(test)]
            fn eq_repr(self, rhs: Self) -> bool {
                if self.is_nan() && rhs.is_nan() {
                    true
                } else {
                    self.repr() == rhs.repr()
                }
            }
            fn from_repr(a: Self::Int) -> Self {
                unsafe { mem::transmute(a) }
            }
            fn from_parts(sign: bool, exponent: Self::Int, significand: Self::Int) -> Self {
                Self::from_repr(((sign as Self::Int) << (Self::BITS - 1)) |
                    ((exponent << Self::SIGNIFICAND_BITS) & Self::EXPONENT_MASK) |
                    (significand & Self::SIGNIFICAND_MASK))
            }
            fn normalize(significand: Self::Int) -> (i32, Self::Int) {
                let shift = significand.leading_zeros()
                    .wrapping_sub((Self::Int::ONE << Self::SIGNIFICAND_BITS).leading_zeros());
                (1i32.wrapping_sub(shift as i32), significand << shift as Self::Int)
            }
        }
    }
}

float_impl!(f32, u32, i32, 32, 23);
float_impl!(f64, u64, i64, 64, 52);
