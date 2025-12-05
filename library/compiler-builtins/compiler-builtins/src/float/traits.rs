use core::ops;

use crate::int::{DInt, Int, MinInt};

/// Wrapper to extract the integer type half of the float's size
pub type HalfRep<F> = <<F as Float>::Int as DInt>::H;

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
    type Int: Int<OtherSign = Self::SignedInt, Unsigned = Self::Int>;

    /// A int of the same width as the float
    type SignedInt: Int + MinInt<OtherSign = Self::Int, Unsigned = Self::Int>;

    /// An int capable of containing the exponent bits plus a sign bit. This is signed.
    type ExpInt: Int;

    const ZERO: Self;
    const ONE: Self;

    /// The bitwidth of the float type.
    const BITS: u32;

    /// The bitwidth of the significand.
    const SIG_BITS: u32;

    /// The bitwidth of the exponent.
    const EXP_BITS: u32 = Self::BITS - Self::SIG_BITS - 1;

    /// The saturated (maximum bitpattern) value of the exponent, i.e. the infinite
    /// representation.
    ///
    /// This is in the rightmost position, use `EXP_MASK` for the shifted value.
    const EXP_SAT: u32 = (1 << Self::EXP_BITS) - 1;

    /// The exponent bias value.
    const EXP_BIAS: u32 = Self::EXP_SAT >> 1;

    /// A mask for the sign bit.
    const SIGN_MASK: Self::Int;

    /// A mask for the significand.
    const SIG_MASK: Self::Int;

    /// The implicit bit of the float format.
    const IMPLICIT_BIT: Self::Int;

    /// A mask for the exponent.
    const EXP_MASK: Self::Int;

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
            const SIG_BITS: u32 = $significand_bits;

            const SIGN_MASK: Self::Int = 1 << (Self::BITS - 1);
            const SIG_MASK: Self::Int = (1 << Self::SIG_BITS) - 1;
            const IMPLICIT_BIT: Self::Int = 1 << Self::SIG_BITS;
            const EXP_MASK: Self::Int = !(Self::SIGN_MASK | Self::SIG_MASK);

            fn to_bits(self) -> Self::Int {
                self.to_bits()
            }
            fn to_bits_signed(self) -> Self::SignedInt {
                self.to_bits() as Self::SignedInt
            }
            fn eq_repr(self, rhs: Self) -> bool {
                #[cfg(feature = "mangled-names")]
                fn is_nan(x: $ty) -> bool {
                    // When using mangled-names, the "real" compiler-builtins might not have the
                    // necessary builtin (__unordtf2) to test whether `f128` is NaN.
                    // FIXME(f16_f128): Remove once the nightly toolchain has the __unordtf2 builtin
                    // x is NaN if all the bits of the exponent are set and the significand is non-0
                    x.to_bits() & $ty::EXP_MASK == $ty::EXP_MASK && x.to_bits() & $ty::SIG_MASK != 0
                }
                #[cfg(not(feature = "mangled-names"))]
                fn is_nan(x: $ty) -> bool {
                    x.is_nan()
                }
                if is_nan(self) && is_nan(rhs) {
                    true
                } else {
                    self.to_bits() == rhs.to_bits()
                }
            }
            fn is_sign_negative(self) -> bool {
                self.is_sign_negative()
            }
            fn exp(self) -> Self::ExpInt {
                ((self.to_bits() & Self::EXP_MASK) >> Self::SIG_BITS) as Self::ExpInt
            }
            fn frac(self) -> Self::Int {
                self.to_bits() & Self::SIG_MASK
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
                        | ((exponent << Self::SIG_BITS) & Self::EXP_MASK)
                        | (significand & Self::SIG_MASK),
                )
            }
            fn normalize(significand: Self::Int) -> (i32, Self::Int) {
                let shift = significand.leading_zeros().wrapping_sub(Self::EXP_BITS);
                (
                    1i32.wrapping_sub(shift as i32),
                    significand << shift as Self::Int,
                )
            }
            fn is_subnormal(self) -> bool {
                (self.to_bits() & Self::EXP_MASK) == Self::Int::ZERO
            }
        }
    };
}

#[cfg(f16_enabled)]
float_impl!(f16, u16, i16, i8, 16, 10);
float_impl!(f32, u32, i32, i16, 32, 23);
float_impl!(f64, u64, i64, i16, 64, 52);
#[cfg(f128_enabled)]
float_impl!(f128, u128, i128, i16, 128, 112);
