use crate::fmt::{Binary, Debug, Display, LowerExp, LowerHex, Octal, UpperExp, UpperHex};
use crate::hash::Hash;
use crate::iter::{Product, Sum};
use crate::mem;
use crate::num::ParseIntError;
use crate::ops::{Add, AddAssign, Neg, Sub, SubAssign};
use crate::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};
use crate::ops::{Div, DivAssign, Mul, MulAssign, Rem, RemAssign};
use crate::ops::{Shl, ShlAssign, Shr, ShrAssign};
use crate::str::FromStr;

mod sealed {
    #[unstable(
        feature = "sealed_int_traits",
        reason = "can be used to write generic code over primitive integers",
        issue = "none"
    )]
    pub trait Sealed {}
}
use sealed::Sealed;

/// This trait provides methods common to all integer primitives.
///
/// This trait is sealed and cannot be implemented for more types; it is
/// implemented for [`i8`], [`i16`], [`i32`], [`i64`], [`i128`], [`isize`],
/// [`u8`], [`u16`], [`u32`], [`u64`], [`u128`] and [`usize`].
#[unstable(
    feature = "sealed_int_traits",
    reason = "can be used to write generic code over primitive integers",
    issue = "none"
)]
pub trait Int
where
    Self: Copy + Default + Hash + Ord,
    Self: Debug + Display + Binary + Octal + LowerHex + UpperHex + LowerExp + UpperExp,
    Self: FromStr<Err = ParseIntError>,
    Self: Add<Output = Self> + AddAssign,
    Self: Sub<Output = Self> + SubAssign,
    Self: Mul<Output = Self> + MulAssign,
    Self: Div<Output = Self> + DivAssign,
    Self: Rem<Output = Self> + RemAssign,
    Self: Not<Output = Self>,
    Self: BitAnd<Output = Self> + BitAndAssign,
    Self: BitOr<Output = Self> + BitOrAssign,
    Self: BitXor<Output = Self> + BitXorAssign,
    Self: Shl<u32, Output = Self> + ShlAssign<u32>,
    Self: Shr<u32, Output = Self> + ShrAssign<u32>,
    Self: Sum + Product,
    Self: Sealed,
{
    /// A byte array with the same size as the integer type.
    type Bytes;

    /// The smallest value that can be represented by this integer type.
    const MIN: Self;

    /// The largest value that can be represented by this integer type.
    const MAX: Self;

    /// The size of this integer type in bits.
    const BITS: u32;

    /// Converts a string slice in a given base to an integer.
    ///
    /// # Panics
    ///
    /// This function panics if `radix` is not in the range from 2 to 36.
    fn from_str_radix(src: &str, radix: u32) -> Result<Self, ParseIntError>;

    /// Returns the number of ones in the binary representation of `self`.
    fn count_ones(self) -> u32;

    /// Returns the number of zeros in the binary representation of `self`.
    fn count_zeros(self) -> u32;

    /// Returns the number of leading zeros in the binary representation of
    /// `self`.
    fn leading_zeros(self) -> u32;

    /// Returns the number of trailing zeros in the binary representation of
    /// `self`.
    fn trailing_zeros(self) -> u32;

    /// Returns the number of leading ones in the binary representation of
    /// `self`.
    fn leading_ones(self) -> u32;

    /// Returns the number of trailing ones in the binary representation of
    /// `self`.
    fn trailing_ones(self) -> u32;

    /// Shifts the bits to the left by a specified amount, `n`, wrapping the
    /// truncated bits to the end of the resulting integer.
    ///
    /// Please note this isn't the same operation as the `<<` shifting operator!
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn rotate_left(self, n: u32) -> Self;

    /// Shifts the bits to the right by a specified amount, `n`, wrapping the
    /// truncated bits to the beginning of the resulting integer.
    ///
    /// Please note this isn't the same operation as the `>>` shifting operator!
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn rotate_right(self, n: u32) -> Self;

    /// Reverses the byte order of the integer.
    fn swap_bytes(self) -> Self;

    /// Reverses the order of bits in the integer. The least significant bit
    /// becomes the most significant bit, second least-significant bit becomes
    /// second most-significant bit, etc.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn reverse_bits(self) -> Self;

    /// Converts an integer from big endian to the target's endianness.
    ///
    /// On big endian this is a no-op. On little endian the bytes are swapped.
    fn from_be(x: Self) -> Self;

    /// Converts an integer from little endian to the target's endianness.
    ///
    /// On little endian this is a no-op. On big endian the bytes are swapped.
    fn from_le(x: Self) -> Self;

    /// Converts `self` to big endian from the target's endianness.
    ///
    /// On big endian this is a no-op. On little endian the bytes are swapped.
    fn to_be(self) -> Self;

    /// Converts `self` to little endian from the target's endianness.
    ///
    /// On little endian this is a no-op. On big endian the bytes are swapped.
    fn to_le(self) -> Self;

    /// Checked integer addition. Computes `self + rhs`, returning `None` if
    /// overflow occurred.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn checked_add(self, rhs: Self) -> Option<Self>;

    /// Unchecked integer addition. Computes `self + rhs`, assuming overflow
    /// cannot occur. This results in undefined behavior when
    /// `self + rhs > Self::MAX` or `self + rhs < Self::MIN`.
    #[unstable(feature = "unchecked_math", reason = "niche optimization path", issue = "none")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    unsafe fn unchecked_add(self, rhs: Self) -> Self;

    /// Checked integer subtraction. Computes `self - rhs`, returning `None` if
    /// overflow occurred.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn checked_sub(self, rhs: Self) -> Option<Self>;

    /// Unchecked integer subtraction. Computes `self - rhs`, assuming overflow
    /// cannot occur. This results in undefined behavior when
    /// `self - rhs > Self::MAX` or `self - rhs < Self::MIN`.
    #[unstable(feature = "unchecked_math", reason = "niche optimization path", issue = "none")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    unsafe fn unchecked_sub(self, rhs: Self) -> Self;

    /// Checked integer multiplication. Computes `self * rhs`, returning `None`
    /// if overflow occurred.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn checked_mul(self, rhs: Self) -> Option<Self>;

    /// Unchecked integer multiplication. Computes `self * rhs`, assuming
    /// overflow cannot occur. This results in undefined behavior when
    /// `self * rhs > Self::MAX` or `self * rhs < Self::MIN`.
    #[unstable(feature = "unchecked_math", reason = "niche optimization path", issue = "none")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    unsafe fn unchecked_mul(self, rhs: Self) -> Self;

    /// Checked integer division. Computes `self / rhs`, returning `None` if
    /// `rhs == 0` or the division results in overflow.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn checked_div(self, rhs: Self) -> Option<Self>;

    /// Checked Euclidean division. Computes `self.div_euclid(rhs)`, returning
    /// `None` if `rhs == 0` or the division results in overflow.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn checked_div_euclid(self, rhs: Self) -> Option<Self>;

    /// Checked integer remainder. Computes `self % rhs`, returning `None` if
    /// `rhs == 0` or the division results in overflow.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn checked_rem(self, rhs: Self) -> Option<Self>;

    /// Checked Euclidean remainder. Computes `self.rem_euclid(rhs)`, returning
    /// `None` if `rhs == 0` or the division results in overflow.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn checked_rem_euclid(self, rhs: Self) -> Option<Self>;

    /// Checked negation. Computes `-self`, returning `None` if `self == MIN`
    /// for signed integers, and unless `self == 0` for unsigned integers.
    fn checked_neg(self) -> Option<Self>;

    /// Checked shift left. Computes `self << rhs`, returning `None` if `rhs` is
    /// larger than or equal to the number of bits in `self`.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn checked_shl(self, rhs: u32) -> Option<Self>;

    /// Checked shift right. Computes `self >> rhs`, returning `None` if `rhs`
    /// is larger than or equal to the number of bits in `self`.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn checked_shr(self, rhs: u32) -> Option<Self>;

    /// Checked exponentiation. Computes `self.pow(exp)`, returning `None` if
    /// overflow occurred.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn checked_pow(self, exp: u32) -> Option<Self>;

    /// Saturating integer addition. Computes `self + rhs`, saturating at the
    /// numeric bounds instead of overflowing.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn saturating_add(self, rhs: Self) -> Self;

    /// Saturating integer subtraction. Computes `self - rhs`, saturating at the
    /// numeric bounds instead of overflowing.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn saturating_sub(self, rhs: Self) -> Self;

    /// Saturating integer multiplication. Computes `self * rhs`, saturating at
    /// the numeric bounds instead of overflowing.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn saturating_mul(self, rhs: Self) -> Self;

    /// Saturating integer exponentiation. Computes `self.pow(exp)`, saturating
    /// at the numeric bounds instead of overflowing.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn saturating_pow(self, exp: u32) -> Self;

    /// Wrapping (modular) addition. Computes `self + rhs`, wrapping around at
    /// the boundary of the type.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn wrapping_add(self, rhs: Self) -> Self;

    /// Wrapping (modular) subtraction. Computes `self - rhs`, wrapping around
    /// at the boundary of the type.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn wrapping_sub(self, rhs: Self) -> Self;

    /// Wrapping (modular) multiplication. Computes `self * rhs`, wrapping
    /// around at the boundary of the type.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn wrapping_mul(self, rhs: Self) -> Self;

    /// Wrapping (modular) division. Computes `self / rhs`, wrapping around at
    /// the boundary of the type.
    ///
    /// The only case where such wrapping can occur is when one divides
    /// `MIN / -1` on a signed type (where `MIN` is the negative minimal value
    /// for the type); this is equivalent to `-MIN`, a positive value that is
    /// too large to represent in the type. In such a case, this function
    /// returns `MIN` itself.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn wrapping_div(self, rhs: Self) -> Self;

    /// Wrapping Euclidean division. Computes `self.div_euclid(rhs)`, wrapping
    /// around at the boundary of the type.
    ///
    /// Wrapping will only occur in `MIN / -1` on a signed type (where `MIN` is
    /// the negative minimal value for the type). This is equivalent to `-MIN`,
    /// a positive value that is too large to represent in the type. In this
    /// case, this method returns `MIN` itself.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn wrapping_div_euclid(self, rhs: Self) -> Self;

    /// Wrapping (modular) remainder. Computes `self % rhs`, wrapping around at
    /// the boundary of the type.
    ///
    /// Such wrap-around never actually occurs mathematically; implementation
    /// artifacts make `x % y` invalid for `MIN / -1` on a signed type (where
    /// `MIN` is the negative minimal value). In such a case, this function
    /// returns `0`.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn wrapping_rem(self, rhs: Self) -> Self;

    /// Wrapping Euclidean remainder. Computes `self.rem_euclid(rhs)`, wrapping
    /// around at the boundary of the type.
    ///
    /// Wrapping will only occur in `MIN % -1` on a signed type (where `MIN` is
    /// the negative minimal value for the type). In this case, this method
    /// returns 0.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn wrapping_rem_euclid(self, rhs: Self) -> Self;

    /// Wrapping (modular) negation. Computes `-self`, wrapping around at the
    /// boundary of the type.
    ///
    /// For signed types, the only case where such wrapping can occur is when
    /// one negates `MIN` (where `MIN` is the negative minimal value for the
    /// type); this is a positive value that is too large to represent in the
    /// type. In such a case, this function returns `MIN` itself.
    ///
    /// For unsigned types, which do not have negative equivalents, all
    /// applications of this function will wrap (except for `-0`). For values
    /// smaller than the corresponding signed type's maximum the result is the
    /// same as casting the corresponding signed value. Any larger values are
    /// equivalent to `MAX + 1 - (val - MAX - 1)` where `MAX` is the
    /// corresponding signed type's maximum.
    fn wrapping_neg(self) -> Self;

    /// Panic-free bitwise shift-left; yields `self << mask(rhs)`, where `mask`
    /// removes any high-order bits of `rhs` that would cause the shift to
    /// exceed the bitwidth of the type.
    ///
    /// Note that this is *not* the same as a rotate-left; the RHS of a wrapping
    /// shift-left is restricted to the range of the type, rather than the bits
    /// shifted out of the LHS being returned to the other end. The primitive
    /// integer types all implement a [`rotate_left`](Self::rotate_left)
    /// function, which may be what you want instead.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn wrapping_shl(self, rhs: u32) -> Self;

    /// Panic-free bitwise shift-right; yields `self >> mask(rhs)`, where `mask`
    /// removes any high-order bits of `rhs` that would cause the shift to
    /// exceed the bitwidth of the type.
    ///
    /// Note that this is *not* the same as a rotate-right; the RHS of a
    /// wrapping shift-right is restricted to the range of the type, rather than
    /// the bits shifted out of the LHS being returned to the other end. The
    /// primitive integer types all implement a
    /// [`rotate_right`](Self::rotate_right) function, which may be what you
    /// want instead.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn wrapping_shr(self, rhs: u32) -> Self;

    /// Wrapping (modular) exponentiation. Computes `self.pow(exp)`, wrapping
    /// around at the boundary of the type.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn wrapping_pow(self, exp: u32) -> Self;

    /// Calculates `self` + `rhs`
    ///
    /// Returns a tuple of the addition along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would have occurred
    /// then the wrapped value is returned.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn overflowing_add(self, rhs: Self) -> (Self, bool);

    /// Calculates `self` - `rhs`
    ///
    /// Returns a tuple of the subtraction along with a boolean indicating
    /// whether an arithmetic overflow would occur. If an overflow would have
    /// occurred then the wrapped value is returned.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn overflowing_sub(self, rhs: Self) -> (Self, bool);

    /// Calculates the multiplication of `self` and `rhs`.
    ///
    /// Returns a tuple of the multiplication along with a boolean indicating
    /// whether an arithmetic overflow would occur. If an overflow would have
    /// occurred then the wrapped value is returned.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn overflowing_mul(self, rhs: Self) -> (Self, bool);

    /// Calculates the divisor when `self` is divided by `rhs`.
    ///
    /// Returns a tuple of the divisor along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would occur then self
    /// is returned.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn overflowing_div(self, rhs: Self) -> (Self, bool);

    /// Calculates the quotient of Euclidean division `self.div_euclid(rhs)`.
    ///
    /// Returns a tuple of the divisor along with a boolean indicating whether
    /// an arithmetic overflow would occur. If an overflow would occur then
    /// `self` is returned.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn overflowing_div_euclid(self, rhs: Self) -> (Self, bool);

    /// Calculates the remainder when `self` is divided by `rhs`.
    ///
    /// Returns a tuple of the remainder after dividing along with a boolean
    /// indicating whether an arithmetic overflow would occur. If an overflow
    /// would occur then 0 is returned.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn overflowing_rem(self, rhs: Self) -> (Self, bool);

    /// Overflowing Euclidean remainder. Calculates `self.rem_euclid(rhs)`.
    ///
    /// Returns a tuple of the remainder after dividing along with a boolean
    /// indicating whether an arithmetic overflow would occur. If an overflow
    /// would occur then 0 is returned.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn overflowing_rem_euclid(self, rhs: Self) -> (Self, bool);

    /// Negates self, overflowing if this is equal to the minimum value.
    ///
    /// Returns a tuple of the negated version of self along with a boolean
    /// indicating whether an overflow happened.
    ///
    /// For signed types, if `self` is the minimum value (e.g., `i32::MIN` for
    /// values of type `i32`), then the minimum value will be returned again and
    /// `true` will be returned for an overflow happening.
    ///
    /// For unsigned types, returns `!self + 1` using wrapping operations to
    /// return the value that represents the negation of this unsigned value.
    /// Note that for positive unsigned values overflow always occurs, but
    /// negating 0 does not overflow.
    fn overflowing_neg(self) -> (Self, bool);

    /// Shifts self left by `rhs` bits.
    ///
    /// Returns a tuple of the shifted version of self along with a boolean
    /// indicating whether the shift value was larger than or equal to the
    /// number of bits. If the shift value is too large, then value is masked
    /// (N-1) where N is the number of bits, and this value is then used to
    /// perform the shift.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn overflowing_shl(self, rhs: u32) -> (Self, bool);

    /// Shifts self right by `rhs` bits.
    ///
    /// Returns a tuple of the shifted version of self along with a boolean
    /// indicating whether the shift value was larger than or equal to the
    /// number of bits. If the shift value is too large, then value is masked
    /// (N-1) where N is the number of bits, and this value is then used to
    /// perform the shift.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn overflowing_shr(self, rhs: u32) -> (Self, bool);

    /// Raises self to the power of `exp`, using exponentiation by squaring.
    ///
    /// Returns a tuple of the exponentiation along with a bool indicating
    /// whether an overflow happened.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn overflowing_pow(self, exp: u32) -> (Self, bool);

    /// Raises self to the power of `exp`, using exponentiation by squaring.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn pow(self, exp: u32) -> Self;

    /// Calculates the quotient of Euclidean division of `self` by `rhs`.
    ///
    /// This computes the integer `q` such that `self = q * rhs + r`, with
    /// `r = self.rem_euclid(rhs)` and `0 <= r < abs(rhs)`.
    ///
    /// In other words, the result is `self / rhs` rounded to the integer `q`
    /// such that `self >= q * rhs`. If `self > 0`, this is equal to round
    /// towards zero (the default in Rust); if `self < 0`, this is equal to
    /// round towards +/- infinity.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0 or the division results in overflow.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn div_euclid(self, rhs: Self) -> Self;

    /// Calculates the least nonnegative remainder of `self (mod rhs)`.
    ///
    /// This is done as if by the Euclidean division algorithm -- given `r =
    /// self.rem_euclid(rhs)`, `self = rhs * self.div_euclid(rhs) + r`, and `0
    /// <= r < abs(rhs)`.
    ///
    /// # Panics
    ///
    /// This function will panic if `rhs` is 0 or the division results in
    /// overflow.
    #[must_use = "this returns the result of the operation, without modifying the original"]
    fn rem_euclid(self, rhs: Self) -> Self;

    /// Return the memory representation of this integer as a byte array in
    /// big-endian (network) byte order.
    fn to_be_bytes(self) -> Self::Bytes;

    /// Return the memory representation of this integer as a byte array in
    /// little-endian byte order.
    fn to_le_bytes(self) -> Self::Bytes;

    /// Return the memory representation of this integer as a byte array in
    /// native byte order.
    ///
    /// As the target platform's native endianness is used, portable code should
    /// use [`to_be_bytes`] or [`to_le_bytes`], as appropriate, instead.
    ///
    /// [`to_be_bytes`]: Self::to_be_bytes
    /// [`to_le_bytes`]: Self::to_le_bytes
    fn to_ne_bytes(self) -> Self::Bytes;

    /// Return the memory representation of this integer as a byte array in
    /// native byte order.
    ///
    /// [`to_ne_bytes`] should be preferred over this whenever possible.
    ///
    /// [`to_ne_bytes`]: Self::to_ne_bytes
    #[unstable(feature = "num_as_ne_bytes", issue = "76976")]
    fn as_ne_bytes(&self) -> &Self::Bytes;

    /// Create a native endian integer value from its representation as a byte
    /// array in big endian.
    fn from_be_bytes(bytes: Self::Bytes) -> Self;

    /// Create a native endian integer value from its representation as a byte
    /// array in little endian.
    fn from_le_bytes(bytes: Self::Bytes) -> Self;

    /// Create a native endian integer value from its memory representation as a
    /// byte array in native endianness.
    ///
    /// As the target platform's native endianness is used, portable code likely
    /// wants to use [`from_be_bytes`] or [`from_le_bytes`], as appropriate
    /// instead.
    ///
    /// [`from_be_bytes`]: Self::from_be_bytes
    /// [`from_le_bytes`]: Self::from_le_bytes
    fn from_ne_bytes(bytes: Self::Bytes) -> Self;
}

/// This trait provides methods common to all signed integer primitives.
///
/// This trait is sealed and cannot be implemented for more types; it is
/// implemented for [`i8`], [`i16`], [`i32`], [`i64`], [`i128`] and [`isize`].
#[unstable(
    feature = "sealed_int_traits",
    reason = "can be used to write generic code over primitive integers",
    issue = "none"
)]
pub trait SignedInt: Int + Neg<Output = Self> {
    /// An unsigned integer type with the same number of bits as `Self`.
    type Unsigned: UnsignedInt;

    /// Checked absolute value. Computes `self.abs()`, returning `None` if
    /// `self == MIN`.
    fn checked_abs(self) -> Option<Self>;

    /// Saturating integer negation. Computes `-self`, returning `MAX` if `self == MIN`
    /// instead of overflowing.
    fn saturating_neg(self) -> Self;

    /// Saturating absolute value. Computes `self.abs()`, returning `MAX` if `self ==
    /// MIN` instead of overflowing.
    fn saturating_abs(self) -> Self;

    /// Wrapping (modular) absolute value. Computes `self.abs()`, wrapping
    /// around at the boundary of the type.
    ///
    /// The only case where such wrapping can occur is when one takes the
    /// absolute value of the negative minimal value for the type; this is a
    /// positive value that is too large to represent in the type. In such a
    /// case, this function returns `MIN` itself.
    fn wrapping_abs(self) -> Self;

    /// Computes the absolute value of `self` without any wrapping or panicking.
    fn unsigned_abs(self) -> Self::Unsigned;

    /// Computes the absolute value of `self`.
    ///
    /// Returns a tuple of the absolute version of self along with a boolean
    /// indicating whether an overflow happened. If self is the minimum value
    /// then the minimum value will be returned again and true will be returned
    /// for an overflow happening.
    fn overflowing_abs(self) -> (Self, bool);

    /// Computes the absolute value of `self`.
    ///
    /// # Overflow behavior
    ///
    /// The absolute value of `MIN` cannot be represented and attempting to
    /// calculate it will cause an overflow. This means that code in debug mode
    /// will trigger a panic on this case and optimized code will return `MIN`
    /// without a panic.
    fn abs(self) -> Self;

    /// Returns a number representing sign of `self`.
    ///
    ///  - `0` if the number is zero
    ///  - `1` if the number is positive
    ///  - `-1` if the number is negative
    fn signum(self) -> Self;

    /// Returns `true` if `self` is positive and `false` if the number is zero
    /// or negative.
    fn is_positive(self) -> bool;

    /// Returns `true` if `self` is negative and `false` if the number is zero
    /// or positive.
    fn is_negative(self) -> bool;
}

/// This trait provides methods common to all unsigned integer primitives.
///
/// This trait is sealed and cannot be implemented for more types; it is
/// implemented for [`u8`], [`u16`], [`u32`], [`u64`], [`u128`] and [`usize`].
#[unstable(
    feature = "sealed_int_traits",
    reason = "can be used to write generic code over primitive integers",
    issue = "none"
)]
pub trait UnsignedInt: Int {
    /// Returns `true` if and only if `self == 2^k` for some `k`.
    fn is_power_of_two(self) -> bool;

    /// Returns the smallest power of two greater than or equal to `self`.
    ///
    /// When return value overflows (i.e., `self > (1 << (N-1))` for type `uN`),
    /// it panics in debug mode and return value is wrapped to 0 in release mode
    /// (the only situation in which method can return 0).
    fn next_power_of_two(self) -> Self;

    /// Returns the smallest power of two greater than or equal to `n`. If the
    /// next power of two is greater than the type's maximum value, `None` is
    /// returned, otherwise the power of twois wrapped in `Some`.
    fn checked_next_power_of_two(self) -> Option<Self>;

    /// Returns the smallest power of two greater than or equal to `n`. If the
    /// next power of two is greater than the type's maximum value,
    /// the return value is wrapped to `0`.
    fn wrapping_next_power_of_two(self) -> Self;
}

macro_rules! delegate {
    (fn $method:ident($($param:ident: $Param:ty),*) -> $Ret:ty) => {
        #[inline]
        fn $method($($param: $Param),*) -> $Ret {
            Self::$method($($param),*)
        }
    };
    (fn $method:ident(self $(, $param:ident: $Param:ty)*) -> $Ret:ty) => {
        #[inline]
        fn $method(self $(, $param: $Param)*) -> $Ret {
            self.$method($($param),*)
        }
    };
    (unsafe fn $method:ident(self $(, $param:ident: $Param:ty)*) -> $Ret:ty) => {
        #[inline]
        unsafe fn $method(self $(, $param: $Param)*) -> $Ret {
            // SAFETY: the caller must uphold the safety contract for the trait
            // method, which is the same as for the inherent method called here.
            unsafe { self.$method($($param),*) }
        }
    };
}

macro_rules! impl_common {
    ($Int:ty) => {
        #[unstable(
            feature = "sealed_int_traits",
            reason = "can be used to write generic code over primitive integers",
            issue = "none"
        )]
        impl Sealed for $Int {}

        #[unstable(
            feature = "sealed_int_traits",
            reason = "can be used to write generic code over primitive integers",
            issue = "none"
        )]
        impl Int for $Int {
            type Bytes = [u8; mem::size_of::<$Int>()];
            const MIN: Self = Self::MIN;
            const MAX: Self = Self::MAX;
            const BITS: u32 = Self::BITS;
            delegate! { fn from_str_radix(src: &str, radix:u32) -> Result<Self, ParseIntError> }
            delegate! { fn count_ones(self) -> u32 }
            delegate! { fn count_zeros(self) -> u32 }
            delegate! { fn leading_zeros(self) -> u32 }
            delegate! { fn trailing_zeros(self) -> u32 }
            delegate! { fn leading_ones(self) -> u32 }
            delegate! { fn trailing_ones(self) -> u32 }
            delegate! { fn rotate_left(self, n: u32) -> Self }
            delegate! { fn rotate_right(self, n: u32) -> Self }
            delegate! { fn swap_bytes(self) -> Self }
            delegate! { fn reverse_bits(self) -> Self }
            delegate! { fn from_be(x: Self) -> Self }
            delegate! { fn from_le(x: Self) -> Self }
            delegate! { fn to_be(self) -> Self }
            delegate! { fn to_le(self) -> Self }
            delegate! { fn checked_add(self, rhs: Self) -> Option<Self> }
            delegate! { unsafe fn unchecked_add(self, rhs: Self) -> Self }
            delegate! { fn checked_sub(self, rhs: Self) -> Option<Self> }
            delegate! { unsafe fn unchecked_sub(self, rhs: Self) -> Self }
            delegate! { fn checked_mul(self, rhs: Self) -> Option<Self> }
            delegate! { unsafe fn unchecked_mul(self, rhs: Self) -> Self }
            delegate! { fn checked_div(self, rhs: Self) -> Option<Self> }
            delegate! { fn checked_div_euclid(self, rhs: Self) -> Option<Self> }
            delegate! { fn checked_rem(self, rhs: Self) -> Option<Self> }
            delegate! { fn checked_rem_euclid(self, rhs: Self) -> Option<Self> }
            delegate! { fn checked_neg(self) -> Option<Self> }
            delegate! { fn checked_shl(self, rhs: u32) -> Option<Self> }
            delegate! { fn checked_shr(self, rhs: u32) -> Option<Self> }
            delegate! { fn checked_pow(self, exp: u32) -> Option<Self> }
            delegate! { fn saturating_add(self, rhs: Self) -> Self }
            delegate! { fn saturating_sub(self, rhs: Self) -> Self }
            delegate! { fn saturating_mul(self, rhs: Self) -> Self }
            delegate! { fn saturating_pow(self, exp: u32) -> Self }
            delegate! { fn wrapping_add(self, rhs: Self) -> Self }
            delegate! { fn wrapping_sub(self, rhs: Self) -> Self }
            delegate! { fn wrapping_mul(self, rhs: Self) -> Self }
            delegate! { fn wrapping_div(self, rhs: Self) -> Self }
            delegate! { fn wrapping_div_euclid(self, rhs: Self) -> Self }
            delegate! { fn wrapping_rem(self, rhs: Self) -> Self }
            delegate! { fn wrapping_rem_euclid(self, rhs: Self) -> Self }
            delegate! { fn wrapping_neg(self) -> Self }
            delegate! { fn wrapping_shl(self, rhs: u32) -> Self }
            delegate! { fn wrapping_shr(self, rhs: u32) -> Self }
            delegate! { fn wrapping_pow(self, exp: u32) -> Self }
            delegate! { fn overflowing_add(self, rhs: Self) -> (Self, bool) }
            delegate! { fn overflowing_sub(self, rhs: Self) -> (Self, bool) }
            delegate! { fn overflowing_mul(self, rhs: Self) -> (Self, bool) }
            delegate! { fn overflowing_div(self, rhs: Self) -> (Self, bool) }
            delegate! { fn overflowing_div_euclid(self, rhs: Self) -> (Self, bool) }
            delegate! { fn overflowing_rem(self, rhs: Self) -> (Self, bool) }
            delegate! { fn overflowing_rem_euclid(self, rhs: Self) -> (Self, bool) }
            delegate! { fn overflowing_neg(self) -> (Self, bool) }
            delegate! { fn overflowing_shl(self, rhs: u32) -> (Self, bool) }
            delegate! { fn overflowing_shr(self, rhs: u32) -> (Self, bool) }
            delegate! { fn overflowing_pow(self, exp: u32) -> (Self, bool) }
            delegate! { fn pow(self, exp: u32) -> Self }
            delegate! { fn div_euclid(self, rhs: Self) -> Self }
            delegate! { fn rem_euclid(self, rhs: Self) -> Self }
            delegate! { fn to_be_bytes(self) -> Self::Bytes }
            delegate! { fn to_le_bytes(self) -> Self::Bytes }
            delegate! { fn to_ne_bytes(self) -> Self::Bytes }

            #[inline]
            fn as_ne_bytes(&self) -> &Self::Bytes {
                self.as_ne_bytes()
            }

            delegate! { fn from_be_bytes(bytes: Self::Bytes) -> Self }
            delegate! { fn from_le_bytes(bytes: Self::Bytes) -> Self }
            delegate! { fn from_ne_bytes(bytes: Self::Bytes) -> Self }
        }
    };
}

macro_rules! impl_signed_unsigned {
    ($SignedInt:ty, $UnsignedInt:ty) => {
        impl_common! { $SignedInt }

        #[unstable(
            feature = "sealed_int_traits",
            reason = "can be used to write generic code over primitive integers",
            issue = "none"
        )]
        impl SignedInt for $SignedInt {
            type Unsigned = $UnsignedInt;
            delegate! { fn checked_abs(self) -> Option<Self> }
            delegate! { fn saturating_neg(self) -> Self }
            delegate! { fn saturating_abs(self) -> Self }
            delegate! { fn wrapping_abs(self) -> Self }
            delegate! { fn unsigned_abs(self) -> Self::Unsigned }
            delegate! { fn overflowing_abs(self) -> (Self, bool) }
            delegate! { fn abs(self) -> Self }
            delegate! { fn signum(self) -> Self }
            delegate! { fn is_positive(self) -> bool }
            delegate! { fn is_negative(self) -> bool }
        }

        impl_common! { $UnsignedInt }

        #[unstable(
            feature = "sealed_int_traits",
            reason = "can be used to write generic code over primitive integers",
            issue = "none"
        )]
        impl UnsignedInt for $UnsignedInt {
            delegate! { fn is_power_of_two(self) -> bool }
            delegate! { fn next_power_of_two(self) -> Self }
            delegate! { fn checked_next_power_of_two(self) -> Option<Self> }
            delegate! { fn wrapping_next_power_of_two(self) -> Self }
        }
    };
}

impl_signed_unsigned! { i8, u8 }
impl_signed_unsigned! { i16, u16 }
impl_signed_unsigned! { i32, u32 }
impl_signed_unsigned! { i64, u64 }
impl_signed_unsigned! { i128, u128 }
impl_signed_unsigned! { isize, usize }
