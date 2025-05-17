//! Interfaces used throughout this crate.

use std::str::FromStr;
use std::{fmt, ops};

use num::Integer;
use num::bigint::ToBigInt;

use crate::validate::Constants;

/// Integer types.
#[allow(dead_code)] // Some functions only used for testing
pub trait Int:
    Clone
    + Copy
    + fmt::Debug
    + fmt::Display
    + fmt::LowerHex
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Shl<u32, Output = Self>
    + ops::Shr<u32, Output = Self>
    + ops::BitAnd<Output = Self>
    + ops::BitOr<Output = Self>
    + ops::Not<Output = Self>
    + ops::AddAssign
    + ops::BitAndAssign
    + ops::BitOrAssign
    + From<u8>
    + TryFrom<i8>
    + TryFrom<u32, Error: fmt::Debug>
    + TryFrom<u64, Error: fmt::Debug>
    + TryFrom<u128, Error: fmt::Debug>
    + TryInto<u64, Error: fmt::Debug>
    + TryInto<u32, Error: fmt::Debug>
    + ToBigInt
    + PartialOrd
    + Integer
    + Send
    + 'static
{
    type Signed: Int;
    type Bytes: Default + AsMut<[u8]>;

    const BITS: u32;
    const ZERO: Self;
    const ONE: Self;
    const MAX: Self;

    fn to_signed(self) -> Self::Signed;
    fn wrapping_neg(self) -> Self;
    fn trailing_zeros(self) -> u32;

    fn hex(self) -> String {
        format!("{self:x}")
    }
}

macro_rules! impl_int {
    ($($uty:ty, $sty:ty);+) => {
        $(
            impl Int for $uty {
                type Signed = $sty;
                type Bytes = [u8; Self::BITS as usize / 8];
                const BITS: u32 = Self::BITS;
                const ZERO: Self = 0;
                const ONE: Self = 1;
                const MAX: Self = Self::MAX;
                fn to_signed(self) -> Self::Signed {
                    self.try_into().unwrap()
                }
                fn wrapping_neg(self) -> Self {
                    self.wrapping_neg()
                }
                fn trailing_zeros(self) -> u32 {
                    self.trailing_zeros()
                }
            }

            impl Int for $sty {
                type Signed = Self;
                type Bytes = [u8; Self::BITS as usize / 8];
                const BITS: u32 = Self::BITS;
                const ZERO: Self = 0;
                const ONE: Self = 1;
                const MAX: Self = Self::MAX;
                fn to_signed(self) -> Self::Signed {
                    self
                }
                fn wrapping_neg(self) -> Self {
                    self.wrapping_neg()
                }
                fn trailing_zeros(self) -> u32 {
                    self.trailing_zeros()
                }
            }
        )+
    }
}

impl_int!(u16, i16; u32, i32; u64, i64);

/// Floating point types.
pub trait Float:
    Copy + fmt::Debug + fmt::LowerExp + FromStr<Err: fmt::Display> + Sized + Send + 'static
{
    /// Unsigned integer of same width
    type Int: Int<Signed = Self::SInt>;
    type SInt: Int;

    /// Total bits
    const BITS: u32;

    /// (Stored) bits in the mantissa)
    const MAN_BITS: u32;

    /// Bits in the exponent
    const EXP_BITS: u32 = Self::BITS - Self::MAN_BITS - 1;

    /// A saturated exponent (all ones)
    const EXP_SAT: u32 = (1 << Self::EXP_BITS) - 1;

    /// The exponent bias, also its maximum value
    const EXP_BIAS: u32 = Self::EXP_SAT >> 1;

    const MAN_MASK: Self::Int;
    const SIGN_MASK: Self::Int;

    fn from_bits(i: Self::Int) -> Self;
    fn to_bits(self) -> Self::Int;

    /// Rational constants associated with this float type.
    fn constants() -> &'static Constants;

    fn is_sign_negative(self) -> bool {
        (self.to_bits() & Self::SIGN_MASK) > Self::Int::ZERO
    }

    /// Exponent without adjustment for bias.
    fn exponent(self) -> u32 {
        ((self.to_bits() >> Self::MAN_BITS) & Self::EXP_SAT.try_into().unwrap()).try_into().unwrap()
    }

    fn mantissa(self) -> Self::Int {
        self.to_bits() & Self::MAN_MASK
    }
}

macro_rules! impl_float {
    ($($fty:ty, $ity:ty);+) => {
        $(
            impl Float for $fty {
                type Int = $ity;
                type SInt = <Self::Int as Int>::Signed;
                const BITS: u32 = <$ity>::BITS;
                const MAN_BITS: u32 = Self::MANTISSA_DIGITS - 1;
                const MAN_MASK: Self::Int = (Self::Int::ONE << Self::MAN_BITS) - Self::Int::ONE;
                const SIGN_MASK: Self::Int = Self::Int::ONE << (Self::BITS-1);
                fn from_bits(i: Self::Int) -> Self { Self::from_bits(i) }
                fn to_bits(self) -> Self::Int { self.to_bits() }
                fn constants() -> &'static Constants {
                    use std::sync::LazyLock;
                    static CONSTANTS: LazyLock<Constants> = LazyLock::new(Constants::new::<$fty>);
                    &CONSTANTS
                }
            }
        )+
    }
}

impl_float!(f32, u32; f64, u64);

#[cfg(target_has_reliable_f16)]
impl_float!(f16, u16);

/// A test generator. Should provide an iterator that produces unique patterns to parse.
///
/// The iterator needs to provide a `WriteCtx` (could be anything), which is then used to
/// write the string at a later step. This is done separately so that we can reuse string
/// allocations (which otherwise turn out to be a pretty expensive part of these tests).
pub trait Generator<F: Float>: Iterator<Item = Self::WriteCtx> + Send + 'static {
    /// Full display and filtering name
    const NAME: &'static str = Self::SHORT_NAME;

    /// Name for display with the progress bar
    const SHORT_NAME: &'static str;

    /// The context needed to create a test string.
    type WriteCtx: Send;

    /// Number of tests that will be run.
    fn total_tests() -> u64;

    /// Constructor for this test generator.
    fn new() -> Self;

    /// Create a test string given write context, which was produced as a step from the iterator.
    ///
    /// `s` will be provided empty.
    fn write_string(s: &mut String, ctx: Self::WriteCtx);
}

/// For tests that use iterator combinators, it is easier to just to box the iterator than trying
/// to specify its type. This is a shorthand for the usual type.
pub type BoxGenIter<This, F> = Box<dyn Iterator<Item = <This as Generator<F>>::WriteCtx> + Send>;
