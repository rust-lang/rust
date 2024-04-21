macro_rules! int_decl {
    () => {
        /// The smallest value that can be represented by this integer type.
        const MIN: Self;

        /// The largest value that can be represented by this integer type.
        const MAX: Self;

        /// The size of this integer type in bits.
        const BITS: u32;

        /// Returns the number of zeros in the binary representation of `self`.
        fn count_ones(self) -> u32;

        /// Returns the number of zeros in the binary representation of `self`.
        fn count_zeros(self) -> u32;

        /// Returns the number of leading zeros in the binary representation of `self`.
        ///
        /// Depending on what you're doing with the value, you might also be interested in the
        /// [`ilog2`] function which returns a consistent number, even if the type widens.
        fn leading_zeros(self) -> u32;

        /// Returns the number of trailing zeros in the binary representation of `self`.
        fn trailing_zeros(self) -> u32;

        /// Returns the number of leading ones in the binary representation of `self`.
        fn leading_ones(self) -> u32;

        /// Returns the number of trailing ones in the binary representation of `self`.
        fn trailing_ones(self) -> u32;

        /// Shifts the bits to the left by a specified amount, `n`,
        /// wrapping the truncated bits to the end of the resulting integer.
        ///
        /// Please note this isn't the same operation as the `<<` shifting operator!
        fn rotate_left(self, n: u32) -> Self;

        /// Shifts the bits to the right by a specified amount, `n`,
        /// wrapping the truncated bits to the beginning of the resulting
        /// integer.
        ///
        /// Please note this isn't the same operation as the `>>` shifting operator!
        fn rotate_right(self, n: u32) -> Self;

        /// Reverses the byte order of the integer.
        fn swap_bytes(self) -> Self;

        /// Reverses the order of bits in the integer. The least significant bit becomes the most significant bit,
        ///                 second least-significant bit becomes second most-significant bit, etc.
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

        /// Checked integer addition. Computes `self + rhs`, returning `None`
        /// if overflow occurred.
        fn checked_add(self, rhs: Self) -> Option<Self>;

        /// Strict integer addition. Computes `self + rhs`, panicking
        /// if overflow occurred.
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        fn strict_add(self, rhs: Self) -> Self;

        /// Unchecked integer addition. Computes `self + rhs`, assuming overflow
        /// cannot occur.
        ///
        /// Calling `x.unchecked_add(y)` is semantically equivalent to calling
        /// `x.`[`checked_add`]`(y).`[`unwrap_unchecked`]`()`.
        ///
        /// If you're just trying to avoid the panic in debug mode, then **do not**
        /// use this.  Instead, you're looking for [`wrapping_add`].
        ///
        /// # Safety
        ///
        /// This results in undefined behavior when `self + rhs > Self::MAX` or
        /// `self + rhs < Self::MIN`, i.e. when [`checked_add`] would return `None`.
        unsafe fn unchecked_add(self, rhs: Self) -> Self;

        /// Checked integer subtraction. Computes `self - rhs`, returning `None` if
        /// overflow occurred.
        fn checked_sub(self, rhs: Self) -> Option<Self>;

        /// Strict integer subtraction. Computes `self - rhs`, panicking if
        /// overflow occurred.
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        fn strict_sub(self, rhs: Self) -> Self;

        /// Unchecked integer subtraction. Computes `self - rhs`, assuming overflow
        /// cannot occur.
        ///
        /// Calling `x.unchecked_sub(y)` is semantically equivalent to calling
        /// `x.`[`checked_sub`]`(y).`[`unwrap_unchecked`]`()`.
        ///
        /// If you're just trying to avoid the panic in debug mode, then **do not**
        /// use this.  Instead, you're looking for [`wrapping_sub`].
        ///
        /// # Safety
        ///
        /// This results in undefined behavior when `self - rhs > Self::MAX` or
        /// `self - rhs < Self::MIN`, i.e. when [`checked_sub`] would return `None`.
        unsafe fn unchecked_sub(self, rhs: Self) -> Self;

        /// Checked integer multiplication. Computes `self * rhs`, returning `None` if
        /// overflow occurred.
        fn checked_mul(self, rhs: Self) -> Option<Self>;

        /// Strict integer multiplication. Computes `self * rhs`, panicking if
        /// overflow occurred.
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        fn strict_mul(self, rhs: Self) -> Self;

        /// Unchecked integer multiplication. Computes `self * rhs`, assuming overflow
        /// cannot occur.
        ///
        /// Calling `x.unchecked_mul(y)` is semantically equivalent to calling
        /// `x.`[`checked_mul`]`(y).`[`unwrap_unchecked`]`()`.
        ///
        /// If you're just trying to avoid the panic in debug mode, then **do not**
        /// use this.  Instead, you're looking for [`wrapping_mul`].
        ///
        /// # Safety
        ///
        /// This results in undefined behavior when `self * rhs > Self::MAX` or
        /// `self * rhs < Self::MIN`, i.e. when [`checked_mul`] would return `None`.
        unsafe fn unchecked_mul(self, rhs: Self) -> Self;

        /// Checked integer division. Computes `self / rhs`, returning `None` if `rhs == 0`
        /// or the division results in overflow.
        fn checked_div(self, rhs: Self) -> Option<Self>;

        /// Strict integer division. Computes `self / rhs`, panicking
        /// if overflow occurred.
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        fn strict_div(self, rhs: Self) -> Self;

        /// Checked Euclidean division. Computes `self.div_euclid(rhs)`,
        /// returning `None` if `rhs == 0` or the division results in overflow.
        fn checked_div_euclid(self, rhs: Self) -> Option<Self>;

        /// Strict Euclidean division. Computes `self.div_euclid(rhs)`, panicking
        /// if overflow occurred.
        fn strict_div_euclid(self, rhs: Self) -> Self;

        /// Checked integer remainder. Computes `self % rhs`, returning `None` if
        /// `rhs == 0` or the division results in overflow.
        fn checked_rem(self, rhs: Self) -> Option<Self>;

        /// Strict integer remainder. Computes `self % rhs`, panicking if
        /// the division results in overflow.
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        fn strict_rem(self, rhs: Self) -> Self;

        /// Checked Euclidean remainder. Computes `self.rem_euclid(rhs)`, returning `None`
        /// if `rhs == 0` or the division results in overflow.
        fn checked_rem_euclid(self, rhs: Self) -> Option<Self>;

        /// Strict Euclidean remainder. Computes `self.rem_euclid(rhs)`, panicking if
        /// the division results in overflow.
        fn strict_rem_euclid(self, rhs: Self) -> Self;

        /// Checked negation. Computes `-self`, returning `None` if the result is `< MIN`.
        fn checked_neg(self) -> Option<Self>;

        /// Strict negation. Computes `-self`, panicking if the result is `< MIN`.
        fn strict_neg(self) -> Self;

        /// Checked shift left. Computes `self << rhs`, returning `None` if `rhs` is larger
        /// than or equal to the number of bits in `self`.
        fn checked_shl(self, rhs: u32) -> Option<Self>;

        /// Strict shift left. Computes `self << rhs`, panicking if `rhs` is larger
        /// than or equal to the number of bits in `self`.
        fn strict_shl(self, rhs: u32) -> Self;

        /// Unchecked shift left. Computes `self << rhs`, assuming that
        /// `rhs` is less than the number of bits in `self`.
        ///
        /// # Safety
        ///
        /// This results in undefined behavior if `rhs` is larger than
        /// or equal to the number of bits in `self`,
        /// i.e. when [`checked_shl`] would return `None`.
        unsafe fn unchecked_shl(self, rhs: u32) -> Self;

        /// Checked shift right. Computes `self >> rhs`, returning `None` if `rhs` is
        /// larger than or equal to the number of bits in `self`.
        fn checked_shr(self, rhs: u32) -> Option<Self>;

        /// Strict shift right. Computes `self >> rhs`, panicking `rhs` is
        /// larger than or equal to the number of bits in `self`.
        fn strict_shr(self, rhs: u32) -> Self;

        /// Unchecked shift right. Computes `self >> rhs`, assuming that
        /// `rhs` is less than the number of bits in `self`.
        ///
        /// # Safety
        ///
        /// This results in undefined behavior if `rhs` is larger than
        /// or equal to the number of bits in `self`,
        /// i.e. when [`checked_shr`] would return `None`.
        unsafe fn unchecked_shr(self, rhs: u32) -> Self;

        /// Checked exponentiation. Computes `self.pow(exp)`, returning `None` if
        /// overflow occurred.
        fn checked_pow(self, exp: u32) -> Option<Self>;

        /// Strict exponentiation. Computes `self.pow(exp)`, panicking if
        /// overflow occurred.
        fn strict_pow(self, exp: u32) -> Self;

        /// Saturating integer addition. Computes `self + rhs`, saturating at
        /// the numeric bounds instead of overflowing.
        fn saturating_add(self, rhs: Self) -> Self;

        /// Saturating integer subtraction. Computes `self - rhs`, saturating at the
        /// numeric bounds instead of overflowing.
        fn saturating_sub(self, rhs: Self) -> Self;

        /// Saturating integer multiplication. Computes `self * rhs`, saturating at the
        /// numeric bounds instead of overflowing.
        fn saturating_mul(self, rhs: Self) -> Self;

        /// Saturating integer division. Computes `self / rhs`, saturating at the
        /// numeric bounds instead of overflowing.
        fn saturating_div(self, rhs: Self) -> Self;

        /// Saturating integer exponentiation. Computes `self.pow(exp)`,
        /// saturating at the numeric bounds instead of overflowing.
        fn saturating_pow(self, exp: u32) -> Self;

        /// Wrapping (modular) addition. Computes `self + rhs`, wrapping around at the
        /// boundary of the type.
        fn wrapping_add(self, rhs: Self) -> Self;

        /// Wrapping (modular) subtraction. Computes `self - rhs`, wrapping around at the
        /// boundary of the type.
        fn wrapping_sub(self, rhs: Self) -> Self;

        /// Wrapping (modular) multiplication. Computes `self * rhs`, wrapping around at
        /// the boundary of the type.
        fn wrapping_mul(self, rhs: Self) -> Self;

        /// Wrapping (modular) division. Computes `self / rhs`, wrapping around at the
        /// boundary of the type.
        ///
        /// The only case where such wrapping can occur is when one divides `MIN / -1` on a signed type (where
        /// `MIN` is the negative minimal value for the type); this is equivalent to `-MIN`, a positive value
        /// that is too large to represent in the type. In such a case, this function returns `MIN` itself.
        fn wrapping_div(self, rhs: Self) -> Self;

        /// Wrapping Euclidean division. Computes `self.div_euclid(rhs)`,
        /// wrapping around at the boundary of the type.
        ///
        /// Wrapping will only occur in `MIN / -1` on a signed type (where `MIN` is the negative minimal value
        /// for the type). This is equivalent to `-MIN`, a positive value that is too large to represent in the
        /// type. In this case, this method returns `MIN` itself.
        fn wrapping_div_euclid(self, rhs: Self) -> Self;

        /// Wrapping (modular) remainder. Computes `self % rhs`, wrapping around at the
        /// boundary of the type.
        ///
        /// Such wrap-around never actually occurs mathematically; implementation artifacts make `x % y`
        /// invalid for `MIN / -1` on a signed type (where `MIN` is the negative minimal value). In such a case,
        /// this function returns `0`.
        fn wrapping_rem(self, rhs: Self) -> Self;

        /// Wrapping Euclidean remainder. Computes `self.rem_euclid(rhs)`, wrapping around
        /// at the boundary of the type.
        ///
        /// Wrapping will only occur in `MIN % -1` on a signed type (where `MIN` is the negative minimal value
        /// for the type). In this case, this method returns 0.
        fn wrapping_rem_euclid(self, rhs: Self) -> Self;

        /// Wrapping (modular) negation. Computes `-self`, wrapping around at the boundary
        /// of the type.
        ///
        /// The only case where such wrapping can occur is when one negates `MIN` on a signed type (where `MIN`
        /// is the negative minimal value for the type); this is a positive value that is too large to represent
        /// in the type. In such a case, this function returns `MIN` itself.
        fn wrapping_neg(self) -> Self;

        /// Panic-free bitwise shift-left; yields `self << mask(rhs)`, where `mask` removes
        /// any high-order bits of `rhs` that would cause the shift to exceed the bitwidth of the type.
        ///
        /// Note that this is *not* the same as a rotate-left; the RHS of a wrapping shift-left is restricted to
        /// the range of the type, rather than the bits shifted out of the LHS being returned to the other end.
        /// The primitive integer types all implement a [`rotate_left`](Self::rotate_left) function,
        /// which may be what you want instead.
        fn wrapping_shl(self, rhs: u32) -> Self;

        /// Panic-free bitwise shift-right; yields `self >> mask(rhs)`,
        /// where `mask` removes any high-order bits of `rhs` that
        /// would cause the shift to exceed the bitwidth of the type.
        ///
        /// Note that this is *not* the same as a rotate-right; the
        /// RHS of a wrapping shift-right is restricted to the range
        /// of the type, rather than the bits shifted out of the LHS
        /// being returned to the other end. The primitive integer
        /// types all implement a [`rotate_right`](Self::rotate_right) function,
        /// which may be what you want instead.
        fn wrapping_shr(self, rhs: u32) -> Self;

        /// Wrapping (modular) exponentiation. Computes `self.pow(exp)`,
        /// wrapping around at the boundary of the type.
        fn wrapping_pow(self, exp: u32) -> Self;

        /// Calculates `self` + `rhs`
        ///
        /// Returns a tuple of the addition along with a boolean indicating
        /// whether an arithmetic overflow would occur. If an overflow would
        /// have occurred then the wrapped value is returned.
        fn overflowing_add(self, rhs: Self) -> (Self, bool);

        /// Calculates `self` - `rhs`
        ///
        /// Returns a tuple of the subtraction along with a boolean indicating
        /// whether an arithmetic overflow would occur. If an overflow would
        /// have occurred then the wrapped value is returned.
        fn overflowing_sub(self, rhs: Self) -> (Self, bool);

        /// Calculates the multiplication of `self` and `rhs`.
        ///
        /// Returns a tuple of the multiplication along with a boolean
        /// indicating whether an arithmetic overflow would occur. If an
        /// overflow would have occurred then the wrapped value is returned.
        fn overflowing_mul(self, rhs: Self) -> (Self, bool);

        /// Calculates the divisor when `self` is divided by `rhs`.
        ///
        /// Returns a tuple of the divisor along with a boolean indicating
        /// whether an arithmetic overflow would occur. If an overflow would
        ///  occur then self is returned.
        fn overflowing_div(self, rhs: Self) -> (Self, bool);

        /// Calculates the quotient of Euclidean division `self.div_euclid(rhs)`.
        ///
        /// Returns a tuple of the divisor along with a boolean indicating
        /// whether an arithmetic overflow would occur. If an overflow would
        ///  occur then self is returned.
        fn overflowing_div_euclid(self, rhs: Self) -> (Self, bool);

        /// Calculates the remainder when `self` is divided by `rhs`.
        ///
        /// Returns a tuple of the remainder after dividing along with a boolean
        /// indicating whether an arithmetic overflow would occur. If an
        /// overflow would occur then 0 is returned.
        fn overflowing_rem(self, rhs: Self) -> (Self, bool);

        /// Overflowing Euclidean remainder. Calculates `self.rem_euclid(rhs)`.
        ///
        /// Returns a tuple of the remainder after dividing along with a boolean
        /// indicating whether an arithmetic overflow would occur. If an
        /// overflow would occur then 0 is returned.
        fn overflowing_rem_euclid(self, rhs: Self) -> (Self, bool);

        /// Negates self, overflowing if this is equal to the minimum value.
        ///
        /// Returns a tuple of the negated version of self along with a boolean
        /// indicating whether an overflow happened. If `self` is the minimum
        /// signed value (e.g., `i32::MIN` for values of type `i32`), then the
        /// minimum value will be returned again and `true` will be returned for
        /// an overflow happening.
        fn overflowing_neg(self) -> (Self, bool);

        /// Shifts self left by `rhs` bits.
        ///
        /// Returns a tuple of the shifted version of self along with a boolean
        /// indicating whether the shift value was larger than or equal to the
        /// number of bits. If the shift value is too large, then value is
        /// masked (N-1) where N is the number of bits, and this value is then
        /// used to perform the shift.
        fn overflowing_shl(self, rhs: u32) -> (Self, bool);

        /// Shifts self right by `rhs` bits.
        ///
        /// Returns a tuple of the shifted version of self along with a boolean
        /// indicating whether the shift value was larger than or equal to the
        /// number of bits. If the shift value is too large, then value is
        /// masked (N-1) where N is the number of bits, and this value is then
        /// used to perform the shift.
        fn overflowing_shr(self, rhs: u32) -> (Self, bool);

        /// Raises self to the power of `exp`, using exponentiation by squaring.
        ///
        /// Returns a tuple of the exponentiation along with a bool indicating
        /// whether an overflow happened.
        fn overflowing_pow(self, exp: u32) -> (Self, bool);

        /// Raises self to the power of `exp`, using exponentiation by squaring.
        fn pow(self, exp: u32) -> Self;

        /// Returns the square root of the number, rounded down.
        #[unstable(feature = "isqrt", issue = "116226")]
        fn isqrt(self) -> Self;

        /// Calculates the quotient of Euclidean division of `self` by `rhs`.
        ///
        /// This computes the integer `q` such that `self = q * rhs + r`, with
        /// `r = self.rem_euclid(rhs)` and `0 <= r < abs(rhs)`.
        ///
        /// In other words, the result is `self / rhs` rounded to the integer `q`
        /// such that `self >= q * rhs`.
        /// If `self > 0`, this is equal to round towards zero (the default in Rust);
        /// if `self < 0`, this is equal to round towards +/- infinity.
        fn div_euclid(self, rhs: Self) -> Self;

        /// Calculates the least nonnegative remainder of `self (mod rhs)`.
        ///
        /// This is done as if by the Euclidean division algorithm -- given
        /// `r = self.rem_euclid(rhs)`, `self = rhs * self.div_euclid(rhs) + r`,
        /// and `0 <= r < abs(rhs)`.
        fn rem_euclid(self, rhs: Self) -> Self;

        /// Calculates the quotient of `self` and `rhs`, rounding the result
        /// towards negative infinity.
        #[unstable(feature = "int_roundings", issue = "88581")]
        fn div_floor(self, rhs: Self) -> Self;

        /// Calculates the quotient of `self` and `rhs`, rounding the result
        /// towards positive infinity.
        #[unstable(feature = "int_roundings", issue = "88581")]
        fn div_ceil(self, rhs: Self) -> Self;

        /// If `rhs` is positive, calculates the smallest value greater than or
        /// equal to `self` that is a multiple of `rhs`. If `rhs` is negative,
        /// calculates the largest value less than or equal to `self` that is a
        /// multiple of `rhs`.
        #[unstable(feature = "int_roundings", issue = "88581")]
        fn next_multiple_of(self, rhs: Self) -> Self;

        /// If `rhs` is positive, calculates the smallest value greater than or
        /// equal to `self` that is a multiple of `rhs`. If `rhs` is negative,
        /// calculates the largest value less than or equal to `self` that is a
        /// multiple of `rhs`. Returns `None` if `rhs` is zero or the operation
        /// would result in overflow.
        #[unstable(feature = "int_roundings", issue = "88581")]
        fn checked_next_multiple_of(self, rhs: Self) -> Option<Self>;

        /// Returns the smallest value that can be represented by this integer type.
        ///
        /// New code should prefer to use [`Self::MIN`] instead.
        #[deprecated(
            since = "TBD",
            note = "replaced by the `MAX` associated constant on this type"
        )]
        fn min_value() -> Self;

        /// Returns the largest value that can be represented by this integer type.
        ///
        /// New code should prefer to use [`Self::MAX`] instead.
        #[deprecated(
            since = "TBD",
            note = "replaced by the `MAX` associated constant on this type"
        )]
        fn max_value() -> Self;
    };
}

macro_rules! int_impl {
    () => {
        const MIN: Self = Self::MIN;
        const MAX: Self = Self::MAX;
        const BITS: u32 = Self::BITS;

        #[inline]
        fn count_ones(self) -> u32 {
            Self::count_ones(self)
        }

        #[inline]
        fn count_zeros(self) -> u32 {
            Self::count_zeros(self)
        }

        #[inline]
        fn leading_zeros(self) -> u32 {
            Self::leading_zeros(self)
        }

        #[inline]
        fn trailing_zeros(self) -> u32 {
            Self::trailing_zeros(self)
        }

        #[inline]
        fn leading_ones(self) -> u32 {
            Self::leading_ones(self)
        }

        #[inline]
        fn trailing_ones(self) -> u32 {
            Self::trailing_ones(self)
        }

        #[inline]
        fn rotate_left(self, n: u32) -> Self {
            Self::rotate_left(self, n)
        }

        #[inline]
        fn rotate_right(self, n: u32) -> Self {
            Self::rotate_right(self, n)
        }

        #[inline]
        fn swap_bytes(self) -> Self {
            Self::swap_bytes(self)
        }

        #[inline]
        fn reverse_bits(self) -> Self {
            Self::reverse_bits(self)
        }

        #[inline]
        fn from_be(x: Self) -> Self {
            Self::from_be(x)
        }

        #[inline]
        fn from_le(x: Self) -> Self {
            Self::from_le(x)
        }

        #[inline]
        fn to_be(self) -> Self {
            Self::to_be(self)
        }

        #[inline]
        fn to_le(self) -> Self {
            Self::to_le(self)
        }

        #[inline]
        fn checked_add(self, rhs: Self) -> Option<Self> {
            Self::checked_add(self, rhs)
        }

        #[inline]
        fn strict_add(self, rhs: Self) -> Self {
            Self::strict_add(self, rhs)
        }

        #[inline]
        unsafe fn unchecked_add(self, rhs: Self) -> Self {
            // SAFETY: deferred to the caller
            unsafe { Self::unchecked_add(self, rhs) }
        }

        #[inline]
        fn checked_sub(self, rhs: Self) -> Option<Self> {
            Self::checked_sub(self, rhs)
        }

        #[inline]
        fn strict_sub(self, rhs: Self) -> Self {
            Self::strict_sub(self, rhs)
        }

        #[inline]
        unsafe fn unchecked_sub(self, rhs: Self) -> Self {
            // SAFETY: deferred to the caller
            unsafe { Self::unchecked_sub(self, rhs) }
        }

        #[inline]
        fn checked_mul(self, rhs: Self) -> Option<Self> {
            Self::checked_mul(self, rhs)
        }

        #[inline]
        fn strict_mul(self, rhs: Self) -> Self {
            Self::strict_mul(self, rhs)
        }

        #[inline]
        unsafe fn unchecked_mul(self, rhs: Self) -> Self {
            // SAFETY: deferred to the caller
            unsafe { Self::unchecked_mul(self, rhs) }
        }

        #[inline]
        fn checked_div(self, rhs: Self) -> Option<Self> {
            Self::checked_div(self, rhs)
        }

        #[inline]
        fn strict_div(self, rhs: Self) -> Self {
            Self::strict_div(self, rhs)
        }

        #[inline]
        fn checked_div_euclid(self, rhs: Self) -> Option<Self> {
            Self::checked_div_euclid(self, rhs)
        }

        #[inline]
        fn strict_div_euclid(self, rhs: Self) -> Self {
            Self::strict_div_euclid(self, rhs)
        }

        #[inline]
        fn checked_rem(self, rhs: Self) -> Option<Self> {
            Self::checked_rem(self, rhs)
        }

        #[inline]
        fn strict_rem(self, rhs: Self) -> Self {
            Self::strict_rem(self, rhs)
        }

        #[inline]
        fn checked_rem_euclid(self, rhs: Self) -> Option<Self> {
            Self::checked_rem_euclid(self, rhs)
        }

        #[inline]
        fn strict_rem_euclid(self, rhs: Self) -> Self {
            Self::strict_rem_euclid(self, rhs)
        }

        #[inline]
        fn checked_neg(self) -> Option<Self> {
            Self::checked_neg(self)
        }

        #[inline]
        fn strict_neg(self) -> Self {
            Self::strict_neg(self)
        }

        #[inline]
        fn checked_shl(self, rhs: u32) -> Option<Self> {
            Self::checked_shl(self, rhs)
        }

        #[inline]
        fn strict_shl(self, rhs: u32) -> Self {
            Self::strict_shl(self, rhs)
        }

        #[inline]
        unsafe fn unchecked_shl(self, rhs: u32) -> Self {
            // SAFETY: deferred to the caller
            unsafe { Self::unchecked_shl(self, rhs) }
        }

        #[inline]
        fn checked_shr(self, rhs: u32) -> Option<Self> {
            Self::checked_shr(self, rhs)
        }

        #[inline]
        fn strict_shr(self, rhs: u32) -> Self {
            Self::strict_shr(self, rhs)
        }

        #[inline]
        unsafe fn unchecked_shr(self, rhs: u32) -> Self {
            // SAFETY: deferred to the caller
            unsafe { Self::unchecked_shr(self, rhs) }
        }

        #[inline]
        fn checked_pow(self, exp: u32) -> Option<Self> {
            Self::checked_pow(self, exp)
        }

        #[inline]
        fn strict_pow(self, exp: u32) -> Self {
            Self::strict_pow(self, exp)
        }

        #[inline]
        fn saturating_add(self, rhs: Self) -> Self {
            Self::saturating_add(self, rhs)
        }

        #[inline]
        fn saturating_sub(self, rhs: Self) -> Self {
            Self::saturating_sub(self, rhs)
        }

        #[inline]
        fn saturating_mul(self, rhs: Self) -> Self {
            Self::saturating_mul(self, rhs)
        }

        #[inline]
        fn saturating_div(self, rhs: Self) -> Self {
            Self::saturating_div(self, rhs)
        }

        #[inline]
        fn saturating_pow(self, exp: u32) -> Self {
            Self::saturating_pow(self, exp)
        }

        #[inline]
        fn wrapping_add(self, rhs: Self) -> Self {
            Self::wrapping_add(self, rhs)
        }

        #[inline]
        fn wrapping_sub(self, rhs: Self) -> Self {
            Self::wrapping_sub(self, rhs)
        }

        #[inline]
        fn wrapping_mul(self, rhs: Self) -> Self {
            Self::wrapping_mul(self, rhs)
        }

        #[inline]
        fn wrapping_div(self, rhs: Self) -> Self {
            Self::wrapping_div(self, rhs)
        }

        #[inline]
        fn wrapping_div_euclid(self, rhs: Self) -> Self {
            Self::wrapping_div_euclid(self, rhs)
        }

        #[inline]
        fn wrapping_rem(self, rhs: Self) -> Self {
            Self::wrapping_rem(self, rhs)
        }

        #[inline]
        fn wrapping_rem_euclid(self, rhs: Self) -> Self {
            Self::wrapping_rem_euclid(self, rhs)
        }

        #[inline]
        fn wrapping_neg(self) -> Self {
            Self::wrapping_neg(self)
        }

        #[inline]
        fn wrapping_shl(self, rhs: u32) -> Self {
            Self::wrapping_shl(self, rhs)
        }

        #[inline]
        fn wrapping_shr(self, rhs: u32) -> Self {
            Self::wrapping_shr(self, rhs)
        }

        #[inline]
        fn wrapping_pow(self, exp: u32) -> Self {
            Self::wrapping_pow(self, exp)
        }

        #[inline]
        fn overflowing_add(self, rhs: Self) -> (Self, bool) {
            Self::overflowing_add(self, rhs)
        }

        #[inline]
        fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
            Self::overflowing_sub(self, rhs)
        }

        #[inline]
        fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
            Self::overflowing_mul(self, rhs)
        }

        #[inline]
        fn overflowing_div(self, rhs: Self) -> (Self, bool) {
            Self::overflowing_div(self, rhs)
        }

        #[inline]
        fn overflowing_div_euclid(self, rhs: Self) -> (Self, bool) {
            Self::overflowing_div_euclid(self, rhs)
        }

        #[inline]
        fn overflowing_rem(self, rhs: Self) -> (Self, bool) {
            Self::overflowing_rem(self, rhs)
        }

        #[inline]
        fn overflowing_rem_euclid(self, rhs: Self) -> (Self, bool) {
            Self::overflowing_rem_euclid(self, rhs)
        }

        #[inline]
        fn overflowing_neg(self) -> (Self, bool) {
            Self::overflowing_neg(self)
        }

        #[inline]
        fn overflowing_shl(self, rhs: u32) -> (Self, bool) {
            Self::overflowing_shl(self, rhs)
        }

        #[inline]
        fn overflowing_shr(self, rhs: u32) -> (Self, bool) {
            Self::overflowing_shr(self, rhs)
        }

        #[inline]
        fn overflowing_pow(self, exp: u32) -> (Self, bool) {
            Self::overflowing_pow(self, exp)
        }

        #[inline]
        fn pow(self, exp: u32) -> Self {
            Self::pow(self, exp)
        }

        #[inline]
        fn isqrt(self) -> Self {
            Self::isqrt(self)
        }

        #[inline]
        fn div_euclid(self, rhs: Self) -> Self {
            Self::div_euclid(self, rhs)
        }

        #[inline]
        fn rem_euclid(self, rhs: Self) -> Self {
            Self::rem_euclid(self, rhs)
        }

        #[inline]
        fn div_floor(self, rhs: Self) -> Self {
            Self::div_floor(self, rhs)
        }

        #[inline]
        fn div_ceil(self, rhs: Self) -> Self {
            Self::div_ceil(self, rhs)
        }

        #[inline]
        fn next_multiple_of(self, rhs: Self) -> Self {
            Self::next_multiple_of(self, rhs)
        }

        #[inline]
        fn checked_next_multiple_of(self, rhs: Self) -> Option<Self> {
            Self::checked_next_multiple_of(self, rhs)
        }

        #[inline]
        #[allow(deprecated_in_future)]
        fn min_value() -> Self {
            Self::min_value()
        }

        #[inline]
        #[allow(deprecated_in_future)]
        fn max_value() -> Self {
            Self::max_value()
        }
    };
}
