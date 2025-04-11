macro_rules! int_impl {
    (
        Self = $SelfT:ty,
        ActualT = $ActualT:ident,
        UnsignedT = $UnsignedT:ty,

        // There are all for use *only* in doc comments.
        // As such, they're all passed as literals -- passing them as a string
        // literal is fine if they need to be multiple code tokens.
        // In non-comments, use the associated constants rather than these.
        BITS = $BITS:literal,
        BITS_MINUS_ONE = $BITS_MINUS_ONE:literal,
        Min = $Min:literal,
        Max = $Max:literal,
        rot = $rot:literal,
        rot_op = $rot_op:literal,
        rot_result = $rot_result:literal,
        swap_op = $swap_op:literal,
        swapped = $swapped:literal,
        reversed = $reversed:literal,
        le_bytes = $le_bytes:literal,
        be_bytes = $be_bytes:literal,
        to_xe_bytes_doc = $to_xe_bytes_doc:expr,
        from_xe_bytes_doc = $from_xe_bytes_doc:expr,
        bound_condition = $bound_condition:literal,
    ) => {
        /// The smallest value that can be represented by this integer type
        #[doc = concat!("(&minus;2<sup>", $BITS_MINUS_ONE, "</sup>", $bound_condition, ").")]
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN, ", stringify!($Min), ");")]
        /// ```
        #[stable(feature = "assoc_int_consts", since = "1.43.0")]
        pub const MIN: Self = !Self::MAX;

        /// The largest value that can be represented by this integer type
        #[doc = concat!("(2<sup>", $BITS_MINUS_ONE, "</sup> &minus; 1", $bound_condition, ").")]
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX, ", stringify!($Max), ");")]
        /// ```
        #[stable(feature = "assoc_int_consts", since = "1.43.0")]
        pub const MAX: Self = (<$UnsignedT>::MAX >> 1) as Self;

        /// The size of this integer type in bits.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::BITS, ", stringify!($BITS), ");")]
        /// ```
        #[stable(feature = "int_bits_const", since = "1.53.0")]
        pub const BITS: u32 = <$UnsignedT>::BITS;

        /// Returns the number of ones in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = 0b100_0000", stringify!($SelfT), ";")]
        ///
        /// assert_eq!(n.count_ones(), 1);
        /// ```
        ///
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[doc(alias = "popcount")]
        #[doc(alias = "popcnt")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn count_ones(self) -> u32 { (self as $UnsignedT).count_ones() }

        /// Returns the number of zeros in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.count_zeros(), 1);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn count_zeros(self) -> u32 {
            (!self).count_ones()
        }

        /// Returns the number of leading zeros in the binary representation of `self`.
        ///
        /// Depending on what you're doing with the value, you might also be interested in the
        /// [`ilog2`] function which returns a consistent number, even if the type widens.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = -1", stringify!($SelfT), ";")]
        ///
        /// assert_eq!(n.leading_zeros(), 0);
        /// ```
        #[doc = concat!("[`ilog2`]: ", stringify!($SelfT), "::ilog2")]
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn leading_zeros(self) -> u32 {
            (self as $UnsignedT).leading_zeros()
        }

        /// Returns the number of trailing zeros in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = -4", stringify!($SelfT), ";")]
        ///
        /// assert_eq!(n.trailing_zeros(), 2);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn trailing_zeros(self) -> u32 {
            (self as $UnsignedT).trailing_zeros()
        }

        /// Returns the number of leading ones in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = -1", stringify!($SelfT), ";")]
        ///
        #[doc = concat!("assert_eq!(n.leading_ones(), ", stringify!($BITS), ");")]
        /// ```
        #[stable(feature = "leading_trailing_ones", since = "1.46.0")]
        #[rustc_const_stable(feature = "leading_trailing_ones", since = "1.46.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn leading_ones(self) -> u32 {
            (self as $UnsignedT).leading_ones()
        }

        /// Returns the number of trailing ones in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = 3", stringify!($SelfT), ";")]
        ///
        /// assert_eq!(n.trailing_ones(), 2);
        /// ```
        #[stable(feature = "leading_trailing_ones", since = "1.46.0")]
        #[rustc_const_stable(feature = "leading_trailing_ones", since = "1.46.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn trailing_ones(self) -> u32 {
            (self as $UnsignedT).trailing_ones()
        }

        /// Returns `self` with only the most significant bit set, or `0` if
        /// the input is `0`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(isolate_most_least_significant_one)]
        ///
        #[doc = concat!("let n: ", stringify!($SelfT), " = 0b_01100100;")]
        ///
        /// assert_eq!(n.isolate_most_significant_one(), 0b_01000000);
        #[doc = concat!("assert_eq!(0_", stringify!($SelfT), ".isolate_most_significant_one(), 0);")]
        /// ```
        #[unstable(feature = "isolate_most_least_significant_one", issue = "136909")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn isolate_most_significant_one(self) -> Self {
            self & (((1 as $SelfT) << (<$SelfT>::BITS - 1)).wrapping_shr(self.leading_zeros()))
        }

        /// Returns `self` with only the least significant bit set, or `0` if
        /// the input is `0`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(isolate_most_least_significant_one)]
        ///
        #[doc = concat!("let n: ", stringify!($SelfT), " = 0b_01100100;")]
        ///
        /// assert_eq!(n.isolate_least_significant_one(), 0b_00000100);
        #[doc = concat!("assert_eq!(0_", stringify!($SelfT), ".isolate_least_significant_one(), 0);")]
        /// ```
        #[unstable(feature = "isolate_most_least_significant_one", issue = "136909")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn isolate_least_significant_one(self) -> Self {
            self & self.wrapping_neg()
        }

        /// Returns the bit pattern of `self` reinterpreted as an unsigned integer of the same size.
        ///
        /// This produces the same result as an `as` cast, but ensures that the bit-width remains
        /// the same.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        ///
        #[doc = concat!("let n = -1", stringify!($SelfT), ";")]
        ///
        #[doc = concat!("assert_eq!(n.cast_unsigned(), ", stringify!($UnsignedT), "::MAX);")]
        /// ```
        #[stable(feature = "integer_sign_cast", since = "1.87.0")]
        #[rustc_const_stable(feature = "integer_sign_cast", since = "1.87.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn cast_unsigned(self) -> $UnsignedT {
            self as $UnsignedT
        }

        /// Shifts the bits to the left by a specified amount, `n`,
        /// wrapping the truncated bits to the end of the resulting integer.
        ///
        /// Please note this isn't the same operation as the `<<` shifting operator!
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = ", $rot_op, stringify!($SelfT), ";")]
        #[doc = concat!("let m = ", $rot_result, ";")]
        ///
        #[doc = concat!("assert_eq!(n.rotate_left(", $rot, "), m);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn rotate_left(self, n: u32) -> Self {
            (self as $UnsignedT).rotate_left(n) as Self
        }

        /// Shifts the bits to the right by a specified amount, `n`,
        /// wrapping the truncated bits to the beginning of the resulting
        /// integer.
        ///
        /// Please note this isn't the same operation as the `>>` shifting operator!
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = ", $rot_result, stringify!($SelfT), ";")]
        #[doc = concat!("let m = ", $rot_op, ";")]
        ///
        #[doc = concat!("assert_eq!(n.rotate_right(", $rot, "), m);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn rotate_right(self, n: u32) -> Self {
            (self as $UnsignedT).rotate_right(n) as Self
        }

        /// Reverses the byte order of the integer.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = ", $swap_op, stringify!($SelfT), ";")]
        ///
        /// let m = n.swap_bytes();
        ///
        #[doc = concat!("assert_eq!(m, ", $swapped, ");")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn swap_bytes(self) -> Self {
            (self as $UnsignedT).swap_bytes() as Self
        }

        /// Reverses the order of bits in the integer. The least significant bit becomes the most significant bit,
        ///                 second least-significant bit becomes second most-significant bit, etc.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = ", $swap_op, stringify!($SelfT), ";")]
        /// let m = n.reverse_bits();
        ///
        #[doc = concat!("assert_eq!(m, ", $reversed, ");")]
        #[doc = concat!("assert_eq!(0, 0", stringify!($SelfT), ".reverse_bits());")]
        /// ```
        #[stable(feature = "reverse_bits", since = "1.37.0")]
        #[rustc_const_stable(feature = "reverse_bits", since = "1.37.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn reverse_bits(self) -> Self {
            (self as $UnsignedT).reverse_bits() as Self
        }

        /// Converts an integer from big endian to the target's endianness.
        ///
        /// On big endian this is a no-op. On little endian the bytes are swapped.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = 0x1A", stringify!($SelfT), ";")]
        ///
        /// if cfg!(target_endian = "big") {
        #[doc = concat!("    assert_eq!(", stringify!($SelfT), "::from_be(n), n)")]
        /// } else {
        #[doc = concat!("    assert_eq!(", stringify!($SelfT), "::from_be(n), n.swap_bytes())")]
        /// }
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_conversions", since = "1.32.0")]
        #[must_use]
        #[inline]
        pub const fn from_be(x: Self) -> Self {
            #[cfg(target_endian = "big")]
            {
                x
            }
            #[cfg(not(target_endian = "big"))]
            {
                x.swap_bytes()
            }
        }

        /// Converts an integer from little endian to the target's endianness.
        ///
        /// On little endian this is a no-op. On big endian the bytes are swapped.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = 0x1A", stringify!($SelfT), ";")]
        ///
        /// if cfg!(target_endian = "little") {
        #[doc = concat!("    assert_eq!(", stringify!($SelfT), "::from_le(n), n)")]
        /// } else {
        #[doc = concat!("    assert_eq!(", stringify!($SelfT), "::from_le(n), n.swap_bytes())")]
        /// }
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_conversions", since = "1.32.0")]
        #[must_use]
        #[inline]
        pub const fn from_le(x: Self) -> Self {
            #[cfg(target_endian = "little")]
            {
                x
            }
            #[cfg(not(target_endian = "little"))]
            {
                x.swap_bytes()
            }
        }

        /// Converts `self` to big endian from the target's endianness.
        ///
        /// On big endian this is a no-op. On little endian the bytes are swapped.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = 0x1A", stringify!($SelfT), ";")]
        ///
        /// if cfg!(target_endian = "big") {
        ///     assert_eq!(n.to_be(), n)
        /// } else {
        ///     assert_eq!(n.to_be(), n.swap_bytes())
        /// }
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_conversions", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn to_be(self) -> Self { // or not to be?
            #[cfg(target_endian = "big")]
            {
                self
            }
            #[cfg(not(target_endian = "big"))]
            {
                self.swap_bytes()
            }
        }

        /// Converts `self` to little endian from the target's endianness.
        ///
        /// On little endian this is a no-op. On big endian the bytes are swapped.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = 0x1A", stringify!($SelfT), ";")]
        ///
        /// if cfg!(target_endian = "little") {
        ///     assert_eq!(n.to_le(), n)
        /// } else {
        ///     assert_eq!(n.to_le(), n.swap_bytes())
        /// }
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_conversions", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn to_le(self) -> Self {
            #[cfg(target_endian = "little")]
            {
                self
            }
            #[cfg(not(target_endian = "little"))]
            {
                self.swap_bytes()
            }
        }

        /// Checked integer addition. Computes `self + rhs`, returning `None`
        /// if overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).checked_add(1), Some(", stringify!($SelfT), "::MAX - 1));")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).checked_add(3), None);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_add(self, rhs: Self) -> Option<Self> {
            let (a, b) = self.overflowing_add(rhs);
            if intrinsics::unlikely(b) { None } else { Some(a) }
        }

        /// Strict integer addition. Computes `self + rhs`, panicking
        /// if overflow occurred.
        ///
        /// # Panics
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).strict_add(1), ", stringify!($SelfT), "::MAX - 1);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = (", stringify!($SelfT), "::MAX - 2).strict_add(3);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_add(self, rhs: Self) -> Self {
            let (a, b) = self.overflowing_add(rhs);
            if b { overflow_panic::add() } else { a }
        }

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
        /// This results in undefined behavior when
        #[doc = concat!("`self + rhs > ", stringify!($SelfT), "::MAX` or `self + rhs < ", stringify!($SelfT), "::MIN`,")]
        /// i.e. when [`checked_add`] would return `None`.
        ///
        /// [`unwrap_unchecked`]: option/enum.Option.html#method.unwrap_unchecked
        #[doc = concat!("[`checked_add`]: ", stringify!($SelfT), "::checked_add")]
        #[doc = concat!("[`wrapping_add`]: ", stringify!($SelfT), "::wrapping_add")]
        #[stable(feature = "unchecked_math", since = "1.79.0")]
        #[rustc_const_stable(feature = "unchecked_math", since = "1.79.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
        pub const unsafe fn unchecked_add(self, rhs: Self) -> Self {
            assert_unsafe_precondition!(
                check_language_ub,
                concat!(stringify!($SelfT), "::unchecked_add cannot overflow"),
                (
                    lhs: $SelfT = self,
                    rhs: $SelfT = rhs,
                ) => !lhs.overflowing_add(rhs).1,
            );

            // SAFETY: this is guaranteed to be safe by the caller.
            unsafe {
                intrinsics::unchecked_add(self, rhs)
            }
        }

        /// Checked addition with an unsigned integer. Computes `self + rhs`,
        /// returning `None` if overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".checked_add_unsigned(2), Some(3));")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).checked_add_unsigned(3), None);")]
        /// ```
        #[stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[rustc_const_stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_add_unsigned(self, rhs: $UnsignedT) -> Option<Self> {
            let (a, b) = self.overflowing_add_unsigned(rhs);
            if intrinsics::unlikely(b) { None } else { Some(a) }
        }

        /// Strict addition with an unsigned integer. Computes `self + rhs`,
        /// panicking if overflow occurred.
        ///
        /// # Panics
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".strict_add_unsigned(2), 3);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = (", stringify!($SelfT), "::MAX - 2).strict_add_unsigned(3);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_add_unsigned(self, rhs: $UnsignedT) -> Self {
            let (a, b) = self.overflowing_add_unsigned(rhs);
            if b { overflow_panic::add() } else { a }
        }

        /// Checked integer subtraction. Computes `self - rhs`, returning `None` if
        /// overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MIN + 2).checked_sub(1), Some(", stringify!($SelfT), "::MIN + 1));")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MIN + 2).checked_sub(3), None);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
            let (a, b) = self.overflowing_sub(rhs);
            if intrinsics::unlikely(b) { None } else { Some(a) }
        }

        /// Strict integer subtraction. Computes `self - rhs`, panicking if
        /// overflow occurred.
        ///
        /// # Panics
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MIN + 2).strict_sub(1), ", stringify!($SelfT), "::MIN + 1);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = (", stringify!($SelfT), "::MIN + 2).strict_sub(3);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_sub(self, rhs: Self) -> Self {
            let (a, b) = self.overflowing_sub(rhs);
            if b { overflow_panic::sub() } else { a }
        }

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
        /// This results in undefined behavior when
        #[doc = concat!("`self - rhs > ", stringify!($SelfT), "::MAX` or `self - rhs < ", stringify!($SelfT), "::MIN`,")]
        /// i.e. when [`checked_sub`] would return `None`.
        ///
        /// [`unwrap_unchecked`]: option/enum.Option.html#method.unwrap_unchecked
        #[doc = concat!("[`checked_sub`]: ", stringify!($SelfT), "::checked_sub")]
        #[doc = concat!("[`wrapping_sub`]: ", stringify!($SelfT), "::wrapping_sub")]
        #[stable(feature = "unchecked_math", since = "1.79.0")]
        #[rustc_const_stable(feature = "unchecked_math", since = "1.79.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
        pub const unsafe fn unchecked_sub(self, rhs: Self) -> Self {
            assert_unsafe_precondition!(
                check_language_ub,
                concat!(stringify!($SelfT), "::unchecked_sub cannot overflow"),
                (
                    lhs: $SelfT = self,
                    rhs: $SelfT = rhs,
                ) => !lhs.overflowing_sub(rhs).1,
            );

            // SAFETY: this is guaranteed to be safe by the caller.
            unsafe {
                intrinsics::unchecked_sub(self, rhs)
            }
        }

        /// Checked subtraction with an unsigned integer. Computes `self - rhs`,
        /// returning `None` if overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".checked_sub_unsigned(2), Some(-1));")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MIN + 2).checked_sub_unsigned(3), None);")]
        /// ```
        #[stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[rustc_const_stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_sub_unsigned(self, rhs: $UnsignedT) -> Option<Self> {
            let (a, b) = self.overflowing_sub_unsigned(rhs);
            if intrinsics::unlikely(b) { None } else { Some(a) }
        }

        /// Strict subtraction with an unsigned integer. Computes `self - rhs`,
        /// panicking if overflow occurred.
        ///
        /// # Panics
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".strict_sub_unsigned(2), -1);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = (", stringify!($SelfT), "::MIN + 2).strict_sub_unsigned(3);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_sub_unsigned(self, rhs: $UnsignedT) -> Self {
            let (a, b) = self.overflowing_sub_unsigned(rhs);
            if b { overflow_panic::sub() } else { a }
        }

        /// Checked integer multiplication. Computes `self * rhs`, returning `None` if
        /// overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.checked_mul(1), Some(", stringify!($SelfT), "::MAX));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.checked_mul(2), None);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
            let (a, b) = self.overflowing_mul(rhs);
            if intrinsics::unlikely(b) { None } else { Some(a) }
        }

        /// Strict integer multiplication. Computes `self * rhs`, panicking if
        /// overflow occurred.
        ///
        /// # Panics
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.strict_mul(1), ", stringify!($SelfT), "::MAX);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ``` should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = ", stringify!($SelfT), "::MAX.strict_mul(2);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_mul(self, rhs: Self) -> Self {
            let (a, b) = self.overflowing_mul(rhs);
            if b { overflow_panic::mul() } else { a }
        }

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
        /// This results in undefined behavior when
        #[doc = concat!("`self * rhs > ", stringify!($SelfT), "::MAX` or `self * rhs < ", stringify!($SelfT), "::MIN`,")]
        /// i.e. when [`checked_mul`] would return `None`.
        ///
        /// [`unwrap_unchecked`]: option/enum.Option.html#method.unwrap_unchecked
        #[doc = concat!("[`checked_mul`]: ", stringify!($SelfT), "::checked_mul")]
        #[doc = concat!("[`wrapping_mul`]: ", stringify!($SelfT), "::wrapping_mul")]
        #[stable(feature = "unchecked_math", since = "1.79.0")]
        #[rustc_const_stable(feature = "unchecked_math", since = "1.79.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
        pub const unsafe fn unchecked_mul(self, rhs: Self) -> Self {
            assert_unsafe_precondition!(
                check_language_ub,
                concat!(stringify!($SelfT), "::unchecked_mul cannot overflow"),
                (
                    lhs: $SelfT = self,
                    rhs: $SelfT = rhs,
                ) => !lhs.overflowing_mul(rhs).1,
            );

            // SAFETY: this is guaranteed to be safe by the caller.
            unsafe {
                intrinsics::unchecked_mul(self, rhs)
            }
        }

        /// Checked integer division. Computes `self / rhs`, returning `None` if `rhs == 0`
        /// or the division results in overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MIN + 1).checked_div(-1), Some(", stringify!($Max), "));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.checked_div(-1), None);")]
        #[doc = concat!("assert_eq!((1", stringify!($SelfT), ").checked_div(0), None);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_checked_int_div", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_div(self, rhs: Self) -> Option<Self> {
            if intrinsics::unlikely(rhs == 0 || ((self == Self::MIN) && (rhs == -1))) {
                None
            } else {
                // SAFETY: div by zero and by INT_MIN have been checked above
                Some(unsafe { intrinsics::unchecked_div(self, rhs) })
            }
        }

        /// Strict integer division. Computes `self / rhs`, panicking
        /// if overflow occurred.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// The only case where such an overflow can occur is when one divides `MIN / -1` on a signed type (where
        /// `MIN` is the negative minimal value for the type); this is equivalent to `-MIN`, a positive value
        /// that is too large to represent in the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MIN + 1).strict_div(-1), ", stringify!($Max), ");")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = ", stringify!($SelfT), "::MIN.strict_div(-1);")]
        /// ```
        ///
        /// The following panics because of division by zero:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = (1", stringify!($SelfT), ").strict_div(0);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_div(self, rhs: Self) -> Self {
            let (a, b) = self.overflowing_div(rhs);
            if b { overflow_panic::div() } else { a }
        }

        /// Checked Euclidean division. Computes `self.div_euclid(rhs)`,
        /// returning `None` if `rhs == 0` or the division results in overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MIN + 1).checked_div_euclid(-1), Some(", stringify!($Max), "));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.checked_div_euclid(-1), None);")]
        #[doc = concat!("assert_eq!((1", stringify!($SelfT), ").checked_div_euclid(0), None);")]
        /// ```
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_div_euclid(self, rhs: Self) -> Option<Self> {
            // Using `&` helps LLVM see that it is the same check made in division.
            if intrinsics::unlikely(rhs == 0 || ((self == Self::MIN) & (rhs == -1))) {
                None
            } else {
                Some(self.div_euclid(rhs))
            }
        }

        /// Strict Euclidean division. Computes `self.div_euclid(rhs)`, panicking
        /// if overflow occurred.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// The only case where such an overflow can occur is when one divides `MIN / -1` on a signed type (where
        /// `MIN` is the negative minimal value for the type); this is equivalent to `-MIN`, a positive value
        /// that is too large to represent in the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MIN + 1).strict_div_euclid(-1), ", stringify!($Max), ");")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = ", stringify!($SelfT), "::MIN.strict_div_euclid(-1);")]
        /// ```
        ///
        /// The following panics because of division by zero:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = (1", stringify!($SelfT), ").strict_div_euclid(0);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_div_euclid(self, rhs: Self) -> Self {
            let (a, b) = self.overflowing_div_euclid(rhs);
            if b { overflow_panic::div() } else { a }
        }

        /// Checked integer remainder. Computes `self % rhs`, returning `None` if
        /// `rhs == 0` or the division results in overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".checked_rem(2), Some(1));")]
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".checked_rem(0), None);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.checked_rem(-1), None);")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_checked_int_div", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_rem(self, rhs: Self) -> Option<Self> {
            if intrinsics::unlikely(rhs == 0 || ((self == Self::MIN) && (rhs == -1))) {
                None
            } else {
                // SAFETY: div by zero and by INT_MIN have been checked above
                Some(unsafe { intrinsics::unchecked_rem(self, rhs) })
            }
        }

        /// Strict integer remainder. Computes `self % rhs`, panicking if
        /// the division results in overflow.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// The only case where such an overflow can occur is `x % y` for `MIN / -1` on a
        /// signed type (where `MIN` is the negative minimal value), which is invalid due to implementation artifacts.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".strict_rem(2), 1);")]
        /// ```
        ///
        /// The following panics because of division by zero:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = 5", stringify!($SelfT), ".strict_rem(0);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = ", stringify!($SelfT), "::MIN.strict_rem(-1);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_rem(self, rhs: Self) -> Self {
            let (a, b) = self.overflowing_rem(rhs);
            if b { overflow_panic::rem() } else { a }
        }

        /// Checked Euclidean remainder. Computes `self.rem_euclid(rhs)`, returning `None`
        /// if `rhs == 0` or the division results in overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".checked_rem_euclid(2), Some(1));")]
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".checked_rem_euclid(0), None);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.checked_rem_euclid(-1), None);")]
        /// ```
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_rem_euclid(self, rhs: Self) -> Option<Self> {
            // Using `&` helps LLVM see that it is the same check made in division.
            if intrinsics::unlikely(rhs == 0 || ((self == Self::MIN) & (rhs == -1))) {
                None
            } else {
                Some(self.rem_euclid(rhs))
            }
        }

        /// Strict Euclidean remainder. Computes `self.rem_euclid(rhs)`, panicking if
        /// the division results in overflow.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// The only case where such an overflow can occur is `x % y` for `MIN / -1` on a
        /// signed type (where `MIN` is the negative minimal value), which is invalid due to implementation artifacts.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".strict_rem_euclid(2), 1);")]
        /// ```
        ///
        /// The following panics because of division by zero:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = 5", stringify!($SelfT), ".strict_rem_euclid(0);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = ", stringify!($SelfT), "::MIN.strict_rem_euclid(-1);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_rem_euclid(self, rhs: Self) -> Self {
            let (a, b) = self.overflowing_rem_euclid(rhs);
            if b { overflow_panic::rem() } else { a }
        }

        /// Checked negation. Computes `-self`, returning `None` if `self == MIN`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".checked_neg(), Some(-5));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.checked_neg(), None);")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_neg(self) -> Option<Self> {
            let (a, b) = self.overflowing_neg();
            if intrinsics::unlikely(b) { None } else { Some(a) }
        }

        /// Unchecked negation. Computes `-self`, assuming overflow cannot occur.
        ///
        /// # Safety
        ///
        /// This results in undefined behavior when
        #[doc = concat!("`self == ", stringify!($SelfT), "::MIN`,")]
        /// i.e. when [`checked_neg`] would return `None`.
        ///
        #[doc = concat!("[`checked_neg`]: ", stringify!($SelfT), "::checked_neg")]
        #[unstable(
            feature = "unchecked_neg",
            reason = "niche optimization path",
            issue = "85122",
        )]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
        pub const unsafe fn unchecked_neg(self) -> Self {
            assert_unsafe_precondition!(
                check_language_ub,
                concat!(stringify!($SelfT), "::unchecked_neg cannot overflow"),
                (
                    lhs: $SelfT = self,
                ) => !lhs.overflowing_neg().1,
            );

            // SAFETY: this is guaranteed to be safe by the caller.
            unsafe {
                intrinsics::unchecked_sub(0, self)
            }
        }

        /// Strict negation. Computes `-self`, panicking if `self == MIN`.
        ///
        /// # Panics
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".strict_neg(), -5);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = ", stringify!($SelfT), "::MIN.strict_neg();")]
        ///
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_neg(self) -> Self {
            let (a, b) = self.overflowing_neg();
            if b { overflow_panic::neg() } else { a }
        }

        /// Checked shift left. Computes `self << rhs`, returning `None` if `rhs` is larger
        /// than or equal to the number of bits in `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0x1", stringify!($SelfT), ".checked_shl(4), Some(0x10));")]
        #[doc = concat!("assert_eq!(0x1", stringify!($SelfT), ".checked_shl(129), None);")]
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".checked_shl(", stringify!($BITS_MINUS_ONE), "), Some(0));")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_shl(self, rhs: u32) -> Option<Self> {
            // Not using overflowing_shl as that's a wrapping shift
            if rhs < Self::BITS {
                // SAFETY: just checked the RHS is in-range
                Some(unsafe { self.unchecked_shl(rhs) })
            } else {
                None
            }
        }

        /// Strict shift left. Computes `self << rhs`, panicking if `rhs` is larger
        /// than or equal to the number of bits in `self`.
        ///
        /// # Panics
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(0x1", stringify!($SelfT), ".strict_shl(4), 0x10);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = 0x1", stringify!($SelfT), ".strict_shl(129);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_shl(self, rhs: u32) -> Self {
            let (a, b) = self.overflowing_shl(rhs);
            if b { overflow_panic::shl() } else { a }
        }

        /// Unchecked shift left. Computes `self << rhs`, assuming that
        /// `rhs` is less than the number of bits in `self`.
        ///
        /// # Safety
        ///
        /// This results in undefined behavior if `rhs` is larger than
        /// or equal to the number of bits in `self`,
        /// i.e. when [`checked_shl`] would return `None`.
        ///
        #[doc = concat!("[`checked_shl`]: ", stringify!($SelfT), "::checked_shl")]
        #[unstable(
            feature = "unchecked_shifts",
            reason = "niche optimization path",
            issue = "85122",
        )]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
        pub const unsafe fn unchecked_shl(self, rhs: u32) -> Self {
            assert_unsafe_precondition!(
                check_language_ub,
                concat!(stringify!($SelfT), "::unchecked_shl cannot overflow"),
                (
                    rhs: u32 = rhs,
                ) => rhs < <$ActualT>::BITS,
            );

            // SAFETY: this is guaranteed to be safe by the caller.
            unsafe {
                intrinsics::unchecked_shl(self, rhs)
            }
        }

        /// Unbounded shift left. Computes `self << rhs`, without bounding the value of `rhs`.
        ///
        /// If `rhs` is larger or equal to the number of bits in `self`,
        /// the entire value is shifted out, and `0` is returned.
        ///
        /// # Examples
        ///
        /// Basic usage:
        /// ```
        #[doc = concat!("assert_eq!(0x1", stringify!($SelfT), ".unbounded_shl(4), 0x10);")]
        #[doc = concat!("assert_eq!(0x1", stringify!($SelfT), ".unbounded_shl(129), 0);")]
        /// ```
        #[stable(feature = "unbounded_shifts", since = "1.87.0")]
        #[rustc_const_stable(feature = "unbounded_shifts", since = "1.87.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn unbounded_shl(self, rhs: u32) -> $SelfT{
            if rhs < Self::BITS {
                // SAFETY:
                // rhs is just checked to be in-range above
                unsafe { self.unchecked_shl(rhs) }
            } else {
                0
            }
        }

        /// Checked shift right. Computes `self >> rhs`, returning `None` if `rhs` is
        /// larger than or equal to the number of bits in `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".checked_shr(4), Some(0x1));")]
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".checked_shr(128), None);")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_shr(self, rhs: u32) -> Option<Self> {
            // Not using overflowing_shr as that's a wrapping shift
            if rhs < Self::BITS {
                // SAFETY: just checked the RHS is in-range
                Some(unsafe { self.unchecked_shr(rhs) })
            } else {
                None
            }
        }

        /// Strict shift right. Computes `self >> rhs`, panicking `rhs` is
        /// larger than or equal to the number of bits in `self`.
        ///
        /// # Panics
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".strict_shr(4), 0x1);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = 0x10", stringify!($SelfT), ".strict_shr(128);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_shr(self, rhs: u32) -> Self {
            let (a, b) = self.overflowing_shr(rhs);
            if b { overflow_panic::shr() } else { a }
        }

        /// Unchecked shift right. Computes `self >> rhs`, assuming that
        /// `rhs` is less than the number of bits in `self`.
        ///
        /// # Safety
        ///
        /// This results in undefined behavior if `rhs` is larger than
        /// or equal to the number of bits in `self`,
        /// i.e. when [`checked_shr`] would return `None`.
        ///
        #[doc = concat!("[`checked_shr`]: ", stringify!($SelfT), "::checked_shr")]
        #[unstable(
            feature = "unchecked_shifts",
            reason = "niche optimization path",
            issue = "85122",
        )]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
        pub const unsafe fn unchecked_shr(self, rhs: u32) -> Self {
            assert_unsafe_precondition!(
                check_language_ub,
                concat!(stringify!($SelfT), "::unchecked_shr cannot overflow"),
                (
                    rhs: u32 = rhs,
                ) => rhs < <$ActualT>::BITS,
            );

            // SAFETY: this is guaranteed to be safe by the caller.
            unsafe {
                intrinsics::unchecked_shr(self, rhs)
            }
        }

        /// Unbounded shift right. Computes `self >> rhs`, without bounding the value of `rhs`.
        ///
        /// If `rhs` is larger or equal to the number of bits in `self`,
        /// the entire value is shifted out, which yields `0` for a positive number,
        /// and `-1` for a negative number.
        ///
        /// # Examples
        ///
        /// Basic usage:
        /// ```
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".unbounded_shr(4), 0x1);")]
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".unbounded_shr(129), 0);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.unbounded_shr(129), -1);")]
        /// ```
        #[stable(feature = "unbounded_shifts", since = "1.87.0")]
        #[rustc_const_stable(feature = "unbounded_shifts", since = "1.87.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn unbounded_shr(self, rhs: u32) -> $SelfT{
            if rhs < Self::BITS {
                // SAFETY:
                // rhs is just checked to be in-range above
                unsafe { self.unchecked_shr(rhs) }
            } else {
                // A shift by `Self::BITS-1` suffices for signed integers, because the sign bit is copied for each of the shifted bits.

                // SAFETY:
                // `Self::BITS-1` is guaranteed to be less than `Self::BITS`
                unsafe { self.unchecked_shr(Self::BITS - 1) }
            }
        }

        /// Checked absolute value. Computes `self.abs()`, returning `None` if
        /// `self == MIN`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!((-5", stringify!($SelfT), ").checked_abs(), Some(5));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.checked_abs(), None);")]
        /// ```
        #[stable(feature = "no_panic_abs", since = "1.13.0")]
        #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_abs(self) -> Option<Self> {
            if self.is_negative() {
                self.checked_neg()
            } else {
                Some(self)
            }
        }

        /// Strict absolute value. Computes `self.abs()`, panicking if
        /// `self == MIN`.
        ///
        /// # Panics
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!((-5", stringify!($SelfT), ").strict_abs(), 5);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = ", stringify!($SelfT), "::MIN.strict_abs();")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_abs(self) -> Self {
            if self.is_negative() {
                self.strict_neg()
            } else {
                self
            }
        }

        /// Checked exponentiation. Computes `self.pow(exp)`, returning `None` if
        /// overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(8", stringify!($SelfT), ".checked_pow(2), Some(64));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.checked_pow(2), None);")]
        /// ```

        #[stable(feature = "no_panic_pow", since = "1.34.0")]
        #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_pow(self, mut exp: u32) -> Option<Self> {
            if exp == 0 {
                return Some(1);
            }
            let mut base = self;
            let mut acc: Self = 1;

            loop {
                if (exp & 1) == 1 {
                    acc = try_opt!(acc.checked_mul(base));
                    // since exp!=0, finally the exp must be 1.
                    if exp == 1 {
                        return Some(acc);
                    }
                }
                exp /= 2;
                base = try_opt!(base.checked_mul(base));
            }
        }

        /// Strict exponentiation. Computes `self.pow(exp)`, panicking if
        /// overflow occurred.
        ///
        /// # Panics
        ///
        /// ## Overflow behavior
        ///
        /// This function will always panic on overflow, regardless of whether overflow checks are enabled.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(8", stringify!($SelfT), ".strict_pow(2), 64);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = ", stringify!($SelfT), "::MAX.strict_pow(2);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_pow(self, mut exp: u32) -> Self {
            if exp == 0 {
                return 1;
            }
            let mut base = self;
            let mut acc: Self = 1;

            loop {
                if (exp & 1) == 1 {
                    acc = acc.strict_mul(base);
                    // since exp!=0, finally the exp must be 1.
                    if exp == 1 {
                        return acc;
                    }
                }
                exp /= 2;
                base = base.strict_mul(base);
            }
        }

        /// Returns the square root of the number, rounded down.
        ///
        /// Returns `None` if `self` is negative.
        ///
        /// # Examples
        ///
        /// Basic usage:
        /// ```
        #[doc = concat!("assert_eq!(10", stringify!($SelfT), ".checked_isqrt(), Some(3));")]
        /// ```
        #[stable(feature = "isqrt", since = "1.84.0")]
        #[rustc_const_stable(feature = "isqrt", since = "1.84.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_isqrt(self) -> Option<Self> {
            if self < 0 {
                None
            } else {
                // SAFETY: Input is nonnegative in this `else` branch.
                let result = unsafe {
                    crate::num::int_sqrt::$ActualT(self as $ActualT) as $SelfT
                };

                // Inform the optimizer what the range of outputs is. If
                // testing `core` crashes with no panic message and a
                // `num::int_sqrt::i*` test failed, it's because your edits
                // caused these assertions to become false.
                //
                // SAFETY: Integer square root is a monotonically nondecreasing
                // function, which means that increasing the input will never
                // cause the output to decrease. Thus, since the input for
                // nonnegative signed integers is bounded by
                // `[0, <$ActualT>::MAX]`, sqrt(n) will be bounded by
                // `[sqrt(0), sqrt(<$ActualT>::MAX)]`.
                unsafe {
                    // SAFETY: `<$ActualT>::MAX` is nonnegative.
                    const MAX_RESULT: $SelfT = unsafe {
                        crate::num::int_sqrt::$ActualT(<$ActualT>::MAX) as $SelfT
                    };

                    crate::hint::assert_unchecked(result >= 0);
                    crate::hint::assert_unchecked(result <= MAX_RESULT);
                }

                Some(result)
            }
        }

        /// Saturating integer addition. Computes `self + rhs`, saturating at the numeric
        /// bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".saturating_add(1), 101);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.saturating_add(100), ", stringify!($SelfT), "::MAX);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.saturating_add(-1), ", stringify!($SelfT), "::MIN);")]
        /// ```

        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_saturating_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn saturating_add(self, rhs: Self) -> Self {
            intrinsics::saturating_add(self, rhs)
        }

        /// Saturating addition with an unsigned integer. Computes `self + rhs`,
        /// saturating at the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".saturating_add_unsigned(2), 3);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.saturating_add_unsigned(100), ", stringify!($SelfT), "::MAX);")]
        /// ```
        #[stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[rustc_const_stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn saturating_add_unsigned(self, rhs: $UnsignedT) -> Self {
            // Overflow can only happen at the upper bound
            // We cannot use `unwrap_or` here because it is not `const`
            match self.checked_add_unsigned(rhs) {
                Some(x) => x,
                None => Self::MAX,
            }
        }

        /// Saturating integer subtraction. Computes `self - rhs`, saturating at the
        /// numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".saturating_sub(127), -27);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.saturating_sub(100), ", stringify!($SelfT), "::MIN);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.saturating_sub(-1), ", stringify!($SelfT), "::MAX);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_saturating_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn saturating_sub(self, rhs: Self) -> Self {
            intrinsics::saturating_sub(self, rhs)
        }

        /// Saturating subtraction with an unsigned integer. Computes `self - rhs`,
        /// saturating at the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".saturating_sub_unsigned(127), -27);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.saturating_sub_unsigned(100), ", stringify!($SelfT), "::MIN);")]
        /// ```
        #[stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[rustc_const_stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn saturating_sub_unsigned(self, rhs: $UnsignedT) -> Self {
            // Overflow can only happen at the lower bound
            // We cannot use `unwrap_or` here because it is not `const`
            match self.checked_sub_unsigned(rhs) {
                Some(x) => x,
                None => Self::MIN,
            }
        }

        /// Saturating integer negation. Computes `-self`, returning `MAX` if `self == MIN`
        /// instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".saturating_neg(), -100);")]
        #[doc = concat!("assert_eq!((-100", stringify!($SelfT), ").saturating_neg(), 100);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.saturating_neg(), ", stringify!($SelfT), "::MAX);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.saturating_neg(), ", stringify!($SelfT), "::MIN + 1);")]
        /// ```

        #[stable(feature = "saturating_neg", since = "1.45.0")]
        #[rustc_const_stable(feature = "const_saturating_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn saturating_neg(self) -> Self {
            intrinsics::saturating_sub(0, self)
        }

        /// Saturating absolute value. Computes `self.abs()`, returning `MAX` if `self ==
        /// MIN` instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".saturating_abs(), 100);")]
        #[doc = concat!("assert_eq!((-100", stringify!($SelfT), ").saturating_abs(), 100);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.saturating_abs(), ", stringify!($SelfT), "::MAX);")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MIN + 1).saturating_abs(), ", stringify!($SelfT), "::MAX);")]
        /// ```

        #[stable(feature = "saturating_neg", since = "1.45.0")]
        #[rustc_const_stable(feature = "const_saturating_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn saturating_abs(self) -> Self {
            if self.is_negative() {
                self.saturating_neg()
            } else {
                self
            }
        }

        /// Saturating integer multiplication. Computes `self * rhs`, saturating at the
        /// numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(10", stringify!($SelfT), ".saturating_mul(12), 120);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.saturating_mul(10), ", stringify!($SelfT), "::MAX);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.saturating_mul(10), ", stringify!($SelfT), "::MIN);")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_saturating_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn saturating_mul(self, rhs: Self) -> Self {
            match self.checked_mul(rhs) {
                Some(x) => x,
                None => if (self < 0) == (rhs < 0) {
                    Self::MAX
                } else {
                    Self::MIN
                }
            }
        }

        /// Saturating integer division. Computes `self / rhs`, saturating at the
        /// numeric bounds instead of overflowing.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".saturating_div(2), 2);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.saturating_div(-1), ", stringify!($SelfT), "::MIN + 1);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.saturating_div(-1), ", stringify!($SelfT), "::MAX);")]
        ///
        /// ```
        #[stable(feature = "saturating_div", since = "1.58.0")]
        #[rustc_const_stable(feature = "saturating_div", since = "1.58.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn saturating_div(self, rhs: Self) -> Self {
            match self.overflowing_div(rhs) {
                (result, false) => result,
                (_result, true) => Self::MAX, // MIN / -1 is the only possible saturating overflow
            }
        }

        /// Saturating integer exponentiation. Computes `self.pow(exp)`,
        /// saturating at the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!((-4", stringify!($SelfT), ").saturating_pow(3), -64);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.saturating_pow(2), ", stringify!($SelfT), "::MAX);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.saturating_pow(3), ", stringify!($SelfT), "::MIN);")]
        /// ```
        #[stable(feature = "no_panic_pow", since = "1.34.0")]
        #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn saturating_pow(self, exp: u32) -> Self {
            match self.checked_pow(exp) {
                Some(x) => x,
                None if self < 0 && exp % 2 == 1 => Self::MIN,
                None => Self::MAX,
            }
        }

        /// Wrapping (modular) addition. Computes `self + rhs`, wrapping around at the
        /// boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".wrapping_add(27), 127);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.wrapping_add(2), ", stringify!($SelfT), "::MIN + 1);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn wrapping_add(self, rhs: Self) -> Self {
            intrinsics::wrapping_add(self, rhs)
        }

        /// Wrapping (modular) addition with an unsigned integer. Computes
        /// `self + rhs`, wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".wrapping_add_unsigned(27), 127);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.wrapping_add_unsigned(2), ", stringify!($SelfT), "::MIN + 1);")]
        /// ```
        #[stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[rustc_const_stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn wrapping_add_unsigned(self, rhs: $UnsignedT) -> Self {
            self.wrapping_add(rhs as Self)
        }

        /// Wrapping (modular) subtraction. Computes `self - rhs`, wrapping around at the
        /// boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".wrapping_sub(127), -127);")]
        #[doc = concat!("assert_eq!((-2", stringify!($SelfT), ").wrapping_sub(", stringify!($SelfT), "::MAX), ", stringify!($SelfT), "::MAX);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn wrapping_sub(self, rhs: Self) -> Self {
            intrinsics::wrapping_sub(self, rhs)
        }

        /// Wrapping (modular) subtraction with an unsigned integer. Computes
        /// `self - rhs`, wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".wrapping_sub_unsigned(127), -127);")]
        #[doc = concat!("assert_eq!((-2", stringify!($SelfT), ").wrapping_sub_unsigned(", stringify!($UnsignedT), "::MAX), -1);")]
        /// ```
        #[stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[rustc_const_stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn wrapping_sub_unsigned(self, rhs: $UnsignedT) -> Self {
            self.wrapping_sub(rhs as Self)
        }

        /// Wrapping (modular) multiplication. Computes `self * rhs`, wrapping around at
        /// the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(10", stringify!($SelfT), ".wrapping_mul(12), 120);")]
        /// assert_eq!(11i8.wrapping_mul(12), -124);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn wrapping_mul(self, rhs: Self) -> Self {
            intrinsics::wrapping_mul(self, rhs)
        }

        /// Wrapping (modular) division. Computes `self / rhs`, wrapping around at the
        /// boundary of the type.
        ///
        /// The only case where such wrapping can occur is when one divides `MIN / -1` on a signed type (where
        /// `MIN` is the negative minimal value for the type); this is equivalent to `-MIN`, a positive value
        /// that is too large to represent in the type. In such a case, this function returns `MIN` itself.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".wrapping_div(10), 10);")]
        /// assert_eq!((-128i8).wrapping_div(-1), -128);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[rustc_const_stable(feature = "const_wrapping_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn wrapping_div(self, rhs: Self) -> Self {
            self.overflowing_div(rhs).0
        }

        /// Wrapping Euclidean division. Computes `self.div_euclid(rhs)`,
        /// wrapping around at the boundary of the type.
        ///
        /// Wrapping will only occur in `MIN / -1` on a signed type (where `MIN` is the negative minimal value
        /// for the type). This is equivalent to `-MIN`, a positive value that is too large to represent in the
        /// type. In this case, this method returns `MIN` itself.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".wrapping_div_euclid(10), 10);")]
        /// assert_eq!((-128i8).wrapping_div_euclid(-1), -128);
        /// ```
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn wrapping_div_euclid(self, rhs: Self) -> Self {
            self.overflowing_div_euclid(rhs).0
        }

        /// Wrapping (modular) remainder. Computes `self % rhs`, wrapping around at the
        /// boundary of the type.
        ///
        /// Such wrap-around never actually occurs mathematically; implementation artifacts make `x % y`
        /// invalid for `MIN / -1` on a signed type (where `MIN` is the negative minimal value). In such a case,
        /// this function returns `0`.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".wrapping_rem(10), 0);")]
        /// assert_eq!((-128i8).wrapping_rem(-1), 0);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[rustc_const_stable(feature = "const_wrapping_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn wrapping_rem(self, rhs: Self) -> Self {
            self.overflowing_rem(rhs).0
        }

        /// Wrapping Euclidean remainder. Computes `self.rem_euclid(rhs)`, wrapping around
        /// at the boundary of the type.
        ///
        /// Wrapping will only occur in `MIN % -1` on a signed type (where `MIN` is the negative minimal value
        /// for the type). In this case, this method returns 0.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".wrapping_rem_euclid(10), 0);")]
        /// assert_eq!((-128i8).wrapping_rem_euclid(-1), 0);
        /// ```
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn wrapping_rem_euclid(self, rhs: Self) -> Self {
            self.overflowing_rem_euclid(rhs).0
        }

        /// Wrapping (modular) negation. Computes `-self`, wrapping around at the boundary
        /// of the type.
        ///
        /// The only case where such wrapping can occur is when one negates `MIN` on a signed type (where `MIN`
        /// is the negative minimal value for the type); this is a positive value that is too large to represent
        /// in the type. In such a case, this function returns `MIN` itself.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".wrapping_neg(), -100);")]
        #[doc = concat!("assert_eq!((-100", stringify!($SelfT), ").wrapping_neg(), 100);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.wrapping_neg(), ", stringify!($SelfT), "::MIN);")]
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn wrapping_neg(self) -> Self {
            (0 as $SelfT).wrapping_sub(self)
        }

        /// Panic-free bitwise shift-left; yields `self << mask(rhs)`, where `mask` removes
        /// any high-order bits of `rhs` that would cause the shift to exceed the bitwidth of the type.
        ///
        /// Note that this is *not* the same as a rotate-left; the RHS of a wrapping shift-left is restricted to
        /// the range of the type, rather than the bits shifted out of the LHS being returned to the other end.
        /// The primitive integer types all implement a [`rotate_left`](Self::rotate_left) function,
        /// which may be what you want instead.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!((-1", stringify!($SelfT), ").wrapping_shl(7), -128);")]
        #[doc = concat!("assert_eq!((-1", stringify!($SelfT), ").wrapping_shl(128), -1);")]
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn wrapping_shl(self, rhs: u32) -> Self {
            // SAFETY: the masking by the bitsize of the type ensures that we do not shift
            // out of bounds
            unsafe {
                self.unchecked_shl(rhs & (Self::BITS - 1))
            }
        }

        /// Panic-free bitwise shift-right; yields `self >> mask(rhs)`, where `mask`
        /// removes any high-order bits of `rhs` that would cause the shift to exceed the bitwidth of the type.
        ///
        /// Note that this is *not* the same as a rotate-right; the RHS of a wrapping shift-right is restricted
        /// to the range of the type, rather than the bits shifted out of the LHS being returned to the other
        /// end. The primitive integer types all implement a [`rotate_right`](Self::rotate_right) function,
        /// which may be what you want instead.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!((-128", stringify!($SelfT), ").wrapping_shr(7), -1);")]
        /// assert_eq!((-128i16).wrapping_shr(64), -128);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn wrapping_shr(self, rhs: u32) -> Self {
            // SAFETY: the masking by the bitsize of the type ensures that we do not shift
            // out of bounds
            unsafe {
                self.unchecked_shr(rhs & (Self::BITS - 1))
            }
        }

        /// Wrapping (modular) absolute value. Computes `self.abs()`, wrapping around at
        /// the boundary of the type.
        ///
        /// The only case where such wrapping can occur is when one takes the absolute value of the negative
        /// minimal value for the type; this is a positive value that is too large to represent in the type. In
        /// such a case, this function returns `MIN` itself.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".wrapping_abs(), 100);")]
        #[doc = concat!("assert_eq!((-100", stringify!($SelfT), ").wrapping_abs(), 100);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.wrapping_abs(), ", stringify!($SelfT), "::MIN);")]
        /// assert_eq!((-128i8).wrapping_abs() as u8, 128);
        /// ```
        #[stable(feature = "no_panic_abs", since = "1.13.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[allow(unused_attributes)]
        #[inline]
        pub const fn wrapping_abs(self) -> Self {
             if self.is_negative() {
                 self.wrapping_neg()
             } else {
                 self
             }
        }

        /// Computes the absolute value of `self` without any wrapping
        /// or panicking.
        ///
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".unsigned_abs(), 100", stringify!($UnsignedT), ");")]
        #[doc = concat!("assert_eq!((-100", stringify!($SelfT), ").unsigned_abs(), 100", stringify!($UnsignedT), ");")]
        /// assert_eq!((-128i8).unsigned_abs(), 128u8);
        /// ```
        #[stable(feature = "unsigned_abs", since = "1.51.0")]
        #[rustc_const_stable(feature = "unsigned_abs", since = "1.51.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn unsigned_abs(self) -> $UnsignedT {
             self.wrapping_abs() as $UnsignedT
        }

        /// Wrapping (modular) exponentiation. Computes `self.pow(exp)`,
        /// wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(3", stringify!($SelfT), ".wrapping_pow(4), 81);")]
        /// assert_eq!(3i8.wrapping_pow(5), -13);
        /// assert_eq!(3i8.wrapping_pow(6), -39);
        /// ```
        #[stable(feature = "no_panic_pow", since = "1.34.0")]
        #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn wrapping_pow(self, mut exp: u32) -> Self {
            if exp == 0 {
                return 1;
            }
            let mut base = self;
            let mut acc: Self = 1;

            if intrinsics::is_val_statically_known(exp) {
                while exp > 1 {
                    if (exp & 1) == 1 {
                        acc = acc.wrapping_mul(base);
                    }
                    exp /= 2;
                    base = base.wrapping_mul(base);
                }

                // since exp!=0, finally the exp must be 1.
                // Deal with the final bit of the exponent separately, since
                // squaring the base afterwards is not necessary.
                acc.wrapping_mul(base)
            } else {
                // This is faster than the above when the exponent is not known
                // at compile time. We can't use the same code for the constant
                // exponent case because LLVM is currently unable to unroll
                // this loop.
                loop {
                    if (exp & 1) == 1 {
                        acc = acc.wrapping_mul(base);
                        // since exp!=0, finally the exp must be 1.
                        if exp == 1 {
                            return acc;
                        }
                    }
                    exp /= 2;
                    base = base.wrapping_mul(base);
                }
            }
        }

        /// Calculates `self` + `rhs`.
        ///
        /// Returns a tuple of the addition along with a boolean indicating
        /// whether an arithmetic overflow would occur. If an overflow would have
        /// occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".overflowing_add(2), (7, false));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.overflowing_add(1), (", stringify!($SelfT), "::MIN, true));")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn overflowing_add(self, rhs: Self) -> (Self, bool) {
            let (a, b) = intrinsics::add_with_overflow(self as $ActualT, rhs as $ActualT);
            (a as Self, b)
        }

        /// Calculates `self` + `rhs` + `carry` and checks for overflow.
        ///
        /// Performs "ternary addition" of two integer operands and a carry-in
        /// bit, and returns a tuple of the sum along with a boolean indicating
        /// whether an arithmetic overflow would occur. On overflow, the wrapped
        /// value is returned.
        ///
        /// This allows chaining together multiple additions to create a wider
        /// addition, and can be useful for bignum addition. This method should
        /// only be used for the most significant word; for the less significant
        /// words the unsigned method
        #[doc = concat!("[`", stringify!($UnsignedT), "::carrying_add`]")]
        /// should be used.
        ///
        /// The output boolean returned by this method is *not* a carry flag,
        /// and should *not* be added to a more significant word.
        ///
        /// If the input carry is false, this method is equivalent to
        /// [`overflowing_add`](Self::overflowing_add).
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// // Only the most significant word is signed.
        /// //
        #[doc = concat!("//   10  MAX    (a = 10 × 2^", stringify!($BITS), " + 2^", stringify!($BITS), " - 1)")]
        #[doc = concat!("// + -5    9    (b = -5 × 2^", stringify!($BITS), " + 9)")]
        /// // ---------
        #[doc = concat!("//    6    8    (sum = 6 × 2^", stringify!($BITS), " + 8)")]
        ///
        #[doc = concat!("let (a1, a0): (", stringify!($SelfT), ", ", stringify!($UnsignedT), ") = (10, ", stringify!($UnsignedT), "::MAX);")]
        #[doc = concat!("let (b1, b0): (", stringify!($SelfT), ", ", stringify!($UnsignedT), ") = (-5, 9);")]
        /// let carry0 = false;
        ///
        #[doc = concat!("// ", stringify!($UnsignedT), "::carrying_add for the less significant words")]
        /// let (sum0, carry1) = a0.carrying_add(b0, carry0);
        /// assert_eq!(carry1, true);
        ///
        #[doc = concat!("// ", stringify!($SelfT), "::carrying_add for the most significant word")]
        /// let (sum1, overflow) = a1.carrying_add(b1, carry1);
        /// assert_eq!(overflow, false);
        ///
        /// assert_eq!((sum1, sum0), (6, 8));
        /// ```
        #[unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn carrying_add(self, rhs: Self, carry: bool) -> (Self, bool) {
            // note: longer-term this should be done via an intrinsic.
            // note: no intermediate overflow is required (https://github.com/rust-lang/rust/issues/85532#issuecomment-1032214946).
            let (a, b) = self.overflowing_add(rhs);
            let (c, d) = a.overflowing_add(carry as $SelfT);
            (c, b != d)
        }

        /// Calculates `self` + `rhs` with an unsigned `rhs`.
        ///
        /// Returns a tuple of the addition along with a boolean indicating
        /// whether an arithmetic overflow would occur. If an overflow would
        /// have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".overflowing_add_unsigned(2), (3, false));")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MIN).overflowing_add_unsigned(", stringify!($UnsignedT), "::MAX), (", stringify!($SelfT), "::MAX, false));")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).overflowing_add_unsigned(3), (", stringify!($SelfT), "::MIN, true));")]
        /// ```
        #[stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[rustc_const_stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn overflowing_add_unsigned(self, rhs: $UnsignedT) -> (Self, bool) {
            let rhs = rhs as Self;
            let (res, overflowed) = self.overflowing_add(rhs);
            (res, overflowed ^ (rhs < 0))
        }

        /// Calculates `self` - `rhs`.
        ///
        /// Returns a tuple of the subtraction along with a boolean indicating whether an arithmetic overflow
        /// would occur. If an overflow would have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".overflowing_sub(2), (3, false));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.overflowing_sub(1), (", stringify!($SelfT), "::MAX, true));")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
            let (a, b) = intrinsics::sub_with_overflow(self as $ActualT, rhs as $ActualT);
            (a as Self, b)
        }

        /// Calculates `self` &minus; `rhs` &minus; `borrow` and checks for
        /// overflow.
        ///
        /// Performs "ternary subtraction" by subtracting both an integer
        /// operand and a borrow-in bit from `self`, and returns a tuple of the
        /// difference along with a boolean indicating whether an arithmetic
        /// overflow would occur. On overflow, the wrapped value is returned.
        ///
        /// This allows chaining together multiple subtractions to create a
        /// wider subtraction, and can be useful for bignum subtraction. This
        /// method should only be used for the most significant word; for the
        /// less significant words the unsigned method
        #[doc = concat!("[`", stringify!($UnsignedT), "::borrowing_sub`]")]
        /// should be used.
        ///
        /// The output boolean returned by this method is *not* a borrow flag,
        /// and should *not* be subtracted from a more significant word.
        ///
        /// If the input borrow is false, this method is equivalent to
        /// [`overflowing_sub`](Self::overflowing_sub).
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// // Only the most significant word is signed.
        /// //
        #[doc = concat!("//    6    8    (a = 6 × 2^", stringify!($BITS), " + 8)")]
        #[doc = concat!("// - -5    9    (b = -5 × 2^", stringify!($BITS), " + 9)")]
        /// // ---------
        #[doc = concat!("//   10  MAX    (diff = 10 × 2^", stringify!($BITS), " + 2^", stringify!($BITS), " - 1)")]
        ///
        #[doc = concat!("let (a1, a0): (", stringify!($SelfT), ", ", stringify!($UnsignedT), ") = (6, 8);")]
        #[doc = concat!("let (b1, b0): (", stringify!($SelfT), ", ", stringify!($UnsignedT), ") = (-5, 9);")]
        /// let borrow0 = false;
        ///
        #[doc = concat!("// ", stringify!($UnsignedT), "::borrowing_sub for the less significant words")]
        /// let (diff0, borrow1) = a0.borrowing_sub(b0, borrow0);
        /// assert_eq!(borrow1, true);
        ///
        #[doc = concat!("// ", stringify!($SelfT), "::borrowing_sub for the most significant word")]
        /// let (diff1, overflow) = a1.borrowing_sub(b1, borrow1);
        /// assert_eq!(overflow, false);
        ///
        #[doc = concat!("assert_eq!((diff1, diff0), (10, ", stringify!($UnsignedT), "::MAX));")]
        /// ```
        #[unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn borrowing_sub(self, rhs: Self, borrow: bool) -> (Self, bool) {
            // note: longer-term this should be done via an intrinsic.
            // note: no intermediate overflow is required (https://github.com/rust-lang/rust/issues/85532#issuecomment-1032214946).
            let (a, b) = self.overflowing_sub(rhs);
            let (c, d) = a.overflowing_sub(borrow as $SelfT);
            (c, b != d)
        }

        /// Calculates `self` - `rhs` with an unsigned `rhs`.
        ///
        /// Returns a tuple of the subtraction along with a boolean indicating
        /// whether an arithmetic overflow would occur. If an overflow would
        /// have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".overflowing_sub_unsigned(2), (-1, false));")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX).overflowing_sub_unsigned(", stringify!($UnsignedT), "::MAX), (", stringify!($SelfT), "::MIN, false));")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MIN + 2).overflowing_sub_unsigned(3), (", stringify!($SelfT), "::MAX, true));")]
        /// ```
        #[stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[rustc_const_stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn overflowing_sub_unsigned(self, rhs: $UnsignedT) -> (Self, bool) {
            let rhs = rhs as Self;
            let (res, overflowed) = self.overflowing_sub(rhs);
            (res, overflowed ^ (rhs < 0))
        }

        /// Calculates the multiplication of `self` and `rhs`.
        ///
        /// Returns a tuple of the multiplication along with a boolean indicating whether an arithmetic overflow
        /// would occur. If an overflow would have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".overflowing_mul(2), (10, false));")]
        /// assert_eq!(1_000_000_000i32.overflowing_mul(10), (1410065408, true));
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
            let (a, b) = intrinsics::mul_with_overflow(self as $ActualT, rhs as $ActualT);
            (a as Self, b)
        }

        /// Calculates the complete product `self * rhs` without the possibility to overflow.
        ///
        /// This returns the low-order (wrapping) bits and the high-order (overflow) bits
        /// of the result as two separate values, in that order.
        ///
        /// If you also need to add a carry to the wide result, then you want
        /// [`Self::carrying_mul`] instead.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// Please note that this example is shared between integer types.
        /// Which explains why `i32` is used here.
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// assert_eq!(5i32.widening_mul(-2), (4294967286, -1));
        /// assert_eq!(1_000_000_000i32.widening_mul(-10), (2884901888, -3));
        /// ```
        #[unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[rustc_const_unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn widening_mul(self, rhs: Self) -> ($UnsignedT, Self) {
            Self::carrying_mul_add(self, rhs, 0, 0)
        }

        /// Calculates the "full multiplication" `self * rhs + carry`
        /// without the possibility to overflow.
        ///
        /// This returns the low-order (wrapping) bits and the high-order (overflow) bits
        /// of the result as two separate values, in that order.
        ///
        /// Performs "long multiplication" which takes in an extra amount to add, and may return an
        /// additional amount of overflow. This allows for chaining together multiple
        /// multiplications to create "big integers" which represent larger values.
        ///
        /// If you don't need the `carry`, then you can use [`Self::widening_mul`] instead.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// Please note that this example is shared between integer types.
        /// Which explains why `i32` is used here.
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// assert_eq!(5i32.carrying_mul(-2, 0), (4294967286, -1));
        /// assert_eq!(5i32.carrying_mul(-2, 10), (0, 0));
        /// assert_eq!(1_000_000_000i32.carrying_mul(-10, 0), (2884901888, -3));
        /// assert_eq!(1_000_000_000i32.carrying_mul(-10, 10), (2884901898, -3));
        #[doc = concat!("assert_eq!(",
            stringify!($SelfT), "::MAX.carrying_mul(", stringify!($SelfT), "::MAX, ", stringify!($SelfT), "::MAX), ",
            "(", stringify!($SelfT), "::MAX.unsigned_abs() + 1, ", stringify!($SelfT), "::MAX / 2));"
        )]
        /// ```
        #[unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[rustc_const_unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn carrying_mul(self, rhs: Self, carry: Self) -> ($UnsignedT, Self) {
            Self::carrying_mul_add(self, rhs, carry, 0)
        }

        /// Calculates the "full multiplication" `self * rhs + carry1 + carry2`
        /// without the possibility to overflow.
        ///
        /// This returns the low-order (wrapping) bits and the high-order (overflow) bits
        /// of the result as two separate values, in that order.
        ///
        /// Performs "long multiplication" which takes in an extra amount to add, and may return an
        /// additional amount of overflow. This allows for chaining together multiple
        /// multiplications to create "big integers" which represent larger values.
        ///
        /// If you don't need either `carry`, then you can use [`Self::widening_mul`] instead,
        /// and if you only need one `carry`, then you can use [`Self::carrying_mul`] instead.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// Please note that this example is shared between integer types.
        /// Which explains why `i32` is used here.
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// assert_eq!(5i32.carrying_mul_add(-2, 0, 0), (4294967286, -1));
        /// assert_eq!(5i32.carrying_mul_add(-2, 10, 10), (10, 0));
        /// assert_eq!(1_000_000_000i32.carrying_mul_add(-10, 0, 0), (2884901888, -3));
        /// assert_eq!(1_000_000_000i32.carrying_mul_add(-10, 10, 10), (2884901908, -3));
        #[doc = concat!("assert_eq!(",
            stringify!($SelfT), "::MAX.carrying_mul_add(", stringify!($SelfT), "::MAX, ", stringify!($SelfT), "::MAX, ", stringify!($SelfT), "::MAX), ",
            "(", stringify!($UnsignedT), "::MAX, ", stringify!($SelfT), "::MAX / 2));"
        )]
        /// ```
        #[unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[rustc_const_unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn carrying_mul_add(self, rhs: Self, carry: Self, add: Self) -> ($UnsignedT, Self) {
            intrinsics::carrying_mul_add(self, rhs, carry, add)
        }

        /// Calculates the divisor when `self` is divided by `rhs`.
        ///
        /// Returns a tuple of the divisor along with a boolean indicating whether an arithmetic overflow would
        /// occur. If an overflow would occur then self is returned.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".overflowing_div(2), (2, false));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.overflowing_div(-1), (", stringify!($SelfT), "::MIN, true));")]
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_overflowing_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        pub const fn overflowing_div(self, rhs: Self) -> (Self, bool) {
            // Using `&` helps LLVM see that it is the same check made in division.
            if intrinsics::unlikely((self == Self::MIN) & (rhs == -1)) {
                (self, true)
            } else {
                (self / rhs, false)
            }
        }

        /// Calculates the quotient of Euclidean division `self.div_euclid(rhs)`.
        ///
        /// Returns a tuple of the divisor along with a boolean indicating whether an arithmetic overflow would
        /// occur. If an overflow would occur then `self` is returned.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".overflowing_div_euclid(2), (2, false));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.overflowing_div_euclid(-1), (", stringify!($SelfT), "::MIN, true));")]
        /// ```
        #[inline]
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        pub const fn overflowing_div_euclid(self, rhs: Self) -> (Self, bool) {
            // Using `&` helps LLVM see that it is the same check made in division.
            if intrinsics::unlikely((self == Self::MIN) & (rhs == -1)) {
                (self, true)
            } else {
                (self.div_euclid(rhs), false)
            }
        }

        /// Calculates the remainder when `self` is divided by `rhs`.
        ///
        /// Returns a tuple of the remainder after dividing along with a boolean indicating whether an
        /// arithmetic overflow would occur. If an overflow would occur then 0 is returned.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".overflowing_rem(2), (1, false));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.overflowing_rem(-1), (0, true));")]
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_overflowing_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        pub const fn overflowing_rem(self, rhs: Self) -> (Self, bool) {
            if intrinsics::unlikely(rhs == -1) {
                (0, self == Self::MIN)
            } else {
                (self % rhs, false)
            }
        }


        /// Overflowing Euclidean remainder. Calculates `self.rem_euclid(rhs)`.
        ///
        /// Returns a tuple of the remainder after dividing along with a boolean indicating whether an
        /// arithmetic overflow would occur. If an overflow would occur then 0 is returned.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".overflowing_rem_euclid(2), (1, false));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.overflowing_rem_euclid(-1), (0, true));")]
        /// ```
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn overflowing_rem_euclid(self, rhs: Self) -> (Self, bool) {
            if intrinsics::unlikely(rhs == -1) {
                (0, self == Self::MIN)
            } else {
                (self.rem_euclid(rhs), false)
            }
        }


        /// Negates self, overflowing if this is equal to the minimum value.
        ///
        /// Returns a tuple of the negated version of self along with a boolean indicating whether an overflow
        /// happened. If `self` is the minimum value (e.g., `i32::MIN` for values of type `i32`), then the
        /// minimum value will be returned again and `true` will be returned for an overflow happening.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(2", stringify!($SelfT), ".overflowing_neg(), (-2, false));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.overflowing_neg(), (", stringify!($SelfT), "::MIN, true));")]
        /// ```
        #[inline]
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[allow(unused_attributes)]
        pub const fn overflowing_neg(self) -> (Self, bool) {
            if intrinsics::unlikely(self == Self::MIN) {
                (Self::MIN, true)
            } else {
                (-self, false)
            }
        }

        /// Shifts self left by `rhs` bits.
        ///
        /// Returns a tuple of the shifted version of self along with a boolean indicating whether the shift
        /// value was larger than or equal to the number of bits. If the shift value is too large, then value is
        /// masked (N-1) where N is the number of bits, and this value is then used to perform the shift.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0x1", stringify!($SelfT),".overflowing_shl(4), (0x10, false));")]
        /// assert_eq!(0x1i32.overflowing_shl(36), (0x10, true));
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".overflowing_shl(", stringify!($BITS_MINUS_ONE), "), (0, false));")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn overflowing_shl(self, rhs: u32) -> (Self, bool) {
            (self.wrapping_shl(rhs), rhs >= Self::BITS)
        }

        /// Shifts self right by `rhs` bits.
        ///
        /// Returns a tuple of the shifted version of self along with a boolean indicating whether the shift
        /// value was larger than or equal to the number of bits. If the shift value is too large, then value is
        /// masked (N-1) where N is the number of bits, and this value is then used to perform the shift.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".overflowing_shr(4), (0x1, false));")]
        /// assert_eq!(0x10i32.overflowing_shr(36), (0x1, true));
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn overflowing_shr(self, rhs: u32) -> (Self, bool) {
            (self.wrapping_shr(rhs), rhs >= Self::BITS)
        }

        /// Computes the absolute value of `self`.
        ///
        /// Returns a tuple of the absolute version of self along with a boolean indicating whether an overflow
        /// happened. If self is the minimum value
        #[doc = concat!("(e.g., ", stringify!($SelfT), "::MIN for values of type ", stringify!($SelfT), "),")]
        /// then the minimum value will be returned again and true will be returned
        /// for an overflow happening.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(10", stringify!($SelfT), ".overflowing_abs(), (10, false));")]
        #[doc = concat!("assert_eq!((-10", stringify!($SelfT), ").overflowing_abs(), (10, false));")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MIN).overflowing_abs(), (", stringify!($SelfT), "::MIN, true));")]
        /// ```
        #[stable(feature = "no_panic_abs", since = "1.13.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn overflowing_abs(self) -> (Self, bool) {
            (self.wrapping_abs(), self == Self::MIN)
        }

        /// Raises self to the power of `exp`, using exponentiation by squaring.
        ///
        /// Returns a tuple of the exponentiation along with a bool indicating
        /// whether an overflow happened.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(3", stringify!($SelfT), ".overflowing_pow(4), (81, false));")]
        /// assert_eq!(3i8.overflowing_pow(5), (-13, true));
        /// ```
        #[stable(feature = "no_panic_pow", since = "1.34.0")]
        #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn overflowing_pow(self, mut exp: u32) -> (Self, bool) {
            if exp == 0 {
                return (1,false);
            }
            let mut base = self;
            let mut acc: Self = 1;
            let mut overflown = false;
            // Scratch space for storing results of overflowing_mul.
            let mut r;

            loop {
                if (exp & 1) == 1 {
                    r = acc.overflowing_mul(base);
                    // since exp!=0, finally the exp must be 1.
                    if exp == 1 {
                        r.1 |= overflown;
                        return r;
                    }
                    acc = r.0;
                    overflown |= r.1;
                }
                exp /= 2;
                r = base.overflowing_mul(base);
                base = r.0;
                overflown |= r.1;
            }
        }

        /// Raises self to the power of `exp`, using exponentiation by squaring.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let x: ", stringify!($SelfT), " = 2; // or any other integer type")]
        ///
        /// assert_eq!(x.pow(5), 32);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[rustc_inherit_overflow_checks]
        pub const fn pow(self, mut exp: u32) -> Self {
            if exp == 0 {
                return 1;
            }
            let mut base = self;
            let mut acc = 1;

            if intrinsics::is_val_statically_known(exp) {
                while exp > 1 {
                    if (exp & 1) == 1 {
                        acc = acc * base;
                    }
                    exp /= 2;
                    base = base * base;
                }

                // since exp!=0, finally the exp must be 1.
                // Deal with the final bit of the exponent separately, since
                // squaring the base afterwards is not necessary and may cause a
                // needless overflow.
                acc * base
            } else {
                // This is faster than the above when the exponent is not known
                // at compile time. We can't use the same code for the constant
                // exponent case because LLVM is currently unable to unroll
                // this loop.
                loop {
                    if (exp & 1) == 1 {
                        acc = acc * base;
                        // since exp!=0, finally the exp must be 1.
                        if exp == 1 {
                            return acc;
                        }
                    }
                    exp /= 2;
                    base = base * base;
                }
            }
        }

        /// Returns the square root of the number, rounded down.
        ///
        /// # Panics
        ///
        /// This function will panic if `self` is negative.
        ///
        /// # Examples
        ///
        /// Basic usage:
        /// ```
        #[doc = concat!("assert_eq!(10", stringify!($SelfT), ".isqrt(), 3);")]
        /// ```
        #[stable(feature = "isqrt", since = "1.84.0")]
        #[rustc_const_stable(feature = "isqrt", since = "1.84.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn isqrt(self) -> Self {
            match self.checked_isqrt() {
                Some(sqrt) => sqrt,
                None => crate::num::int_sqrt::panic_for_negative_argument(),
            }
        }

        /// Calculates the quotient of Euclidean division of `self` by `rhs`.
        ///
        /// This computes the integer `q` such that `self = q * rhs + r`, with
        /// `r = self.rem_euclid(rhs)` and `0 <= r < abs(rhs)`.
        ///
        /// In other words, the result is `self / rhs` rounded to the integer `q`
        /// such that `self >= q * rhs`.
        /// If `self > 0`, this is equal to rounding towards zero (the default in Rust);
        /// if `self < 0`, this is equal to rounding away from zero (towards +/- infinity).
        /// If `rhs > 0`, this is equal to rounding towards -infinity;
        /// if `rhs < 0`, this is equal to rounding towards +infinity.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero or if `self` is `Self::MIN`
        /// and `rhs` is -1. This behavior is not affected by the `overflow-checks` flag.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let a: ", stringify!($SelfT), " = 7; // or any other integer type")]
        /// let b = 4;
        ///
        /// assert_eq!(a.div_euclid(b), 1); // 7 >= 4 * 1
        /// assert_eq!(a.div_euclid(-b), -1); // 7 >= -4 * -1
        /// assert_eq!((-a).div_euclid(b), -2); // -7 >= 4 * -2
        /// assert_eq!((-a).div_euclid(-b), 2); // -7 >= -4 * 2
        /// ```
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn div_euclid(self, rhs: Self) -> Self {
            let q = self / rhs;
            if self % rhs < 0 {
                return if rhs > 0 { q - 1 } else { q + 1 }
            }
            q
        }


        /// Calculates the least nonnegative remainder of `self (mod rhs)`.
        ///
        /// This is done as if by the Euclidean division algorithm -- given
        /// `r = self.rem_euclid(rhs)`, the result satisfies
        /// `self = rhs * self.div_euclid(rhs) + r` and `0 <= r < abs(rhs)`.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero or if `self` is `Self::MIN` and
        /// `rhs` is -1. This behavior is not affected by the `overflow-checks` flag.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let a: ", stringify!($SelfT), " = 7; // or any other integer type")]
        /// let b = 4;
        ///
        /// assert_eq!(a.rem_euclid(b), 3);
        /// assert_eq!((-a).rem_euclid(b), 1);
        /// assert_eq!(a.rem_euclid(-b), 3);
        /// assert_eq!((-a).rem_euclid(-b), 1);
        /// ```
        ///
        /// This will panic:
        /// ```should_panic
        #[doc = concat!("let _ = ", stringify!($SelfT), "::MIN.rem_euclid(-1);")]
        /// ```
        #[doc(alias = "modulo", alias = "mod")]
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn rem_euclid(self, rhs: Self) -> Self {
            let r = self % rhs;
            if r < 0 {
                // Semantically equivalent to `if rhs < 0 { r - rhs } else { r + rhs }`.
                // If `rhs` is not `Self::MIN`, then `r + abs(rhs)` will not overflow
                // and is clearly equivalent, because `r` is negative.
                // Otherwise, `rhs` is `Self::MIN`, then we have
                // `r.wrapping_add(Self::MIN.wrapping_abs())`, which evaluates
                // to `r.wrapping_add(Self::MIN)`, which is equivalent to
                // `r - Self::MIN`, which is what we wanted (and will not overflow
                // for negative `r`).
                r.wrapping_add(rhs.wrapping_abs())
            } else {
                r
            }
        }

        /// Calculates the quotient of `self` and `rhs`, rounding the result towards negative infinity.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero or if `self` is `Self::MIN`
        /// and `rhs` is -1. This behavior is not affected by the `overflow-checks` flag.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(int_roundings)]
        #[doc = concat!("let a: ", stringify!($SelfT)," = 8;")]
        /// let b = 3;
        ///
        /// assert_eq!(a.div_floor(b), 2);
        /// assert_eq!(a.div_floor(-b), -3);
        /// assert_eq!((-a).div_floor(b), -3);
        /// assert_eq!((-a).div_floor(-b), 2);
        /// ```
        #[unstable(feature = "int_roundings", issue = "88581")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn div_floor(self, rhs: Self) -> Self {
            let d = self / rhs;
            let r = self % rhs;

            // If the remainder is non-zero, we need to subtract one if the
            // signs of self and rhs differ, as this means we rounded upwards
            // instead of downwards. We do this branchlessly by creating a mask
            // which is all-ones iff the signs differ, and 0 otherwise. Then by
            // adding this mask (which corresponds to the signed value -1), we
            // get our correction.
            let correction = (self ^ rhs) >> (Self::BITS - 1);
            if r != 0 {
                d + correction
            } else {
                d
            }
        }

        /// Calculates the quotient of `self` and `rhs`, rounding the result towards positive infinity.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero or if `self` is `Self::MIN`
        /// and `rhs` is -1. This behavior is not affected by the `overflow-checks` flag.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(int_roundings)]
        #[doc = concat!("let a: ", stringify!($SelfT)," = 8;")]
        /// let b = 3;
        ///
        /// assert_eq!(a.div_ceil(b), 3);
        /// assert_eq!(a.div_ceil(-b), -2);
        /// assert_eq!((-a).div_ceil(b), -2);
        /// assert_eq!((-a).div_ceil(-b), 3);
        /// ```
        #[unstable(feature = "int_roundings", issue = "88581")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn div_ceil(self, rhs: Self) -> Self {
            let d = self / rhs;
            let r = self % rhs;

            // When remainder is non-zero we have a.div_ceil(b) == 1 + a.div_floor(b),
            // so we can re-use the algorithm from div_floor, just adding 1.
            let correction = 1 + ((self ^ rhs) >> (Self::BITS - 1));
            if r != 0 {
                d + correction
            } else {
                d
            }
        }

        /// If `rhs` is positive, calculates the smallest value greater than or
        /// equal to `self` that is a multiple of `rhs`. If `rhs` is negative,
        /// calculates the largest value less than or equal to `self` that is a
        /// multiple of `rhs`.
        ///
        /// # Panics
        ///
        /// This function will panic if `rhs` is zero.
        ///
        /// ## Overflow behavior
        ///
        /// On overflow, this function will panic if overflow checks are enabled (default in debug
        /// mode) and wrap if overflow checks are disabled (default in release mode).
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(int_roundings)]
        #[doc = concat!("assert_eq!(16_", stringify!($SelfT), ".next_multiple_of(8), 16);")]
        #[doc = concat!("assert_eq!(23_", stringify!($SelfT), ".next_multiple_of(8), 24);")]
        #[doc = concat!("assert_eq!(16_", stringify!($SelfT), ".next_multiple_of(-8), 16);")]
        #[doc = concat!("assert_eq!(23_", stringify!($SelfT), ".next_multiple_of(-8), 16);")]
        #[doc = concat!("assert_eq!((-16_", stringify!($SelfT), ").next_multiple_of(8), -16);")]
        #[doc = concat!("assert_eq!((-23_", stringify!($SelfT), ").next_multiple_of(8), -16);")]
        #[doc = concat!("assert_eq!((-16_", stringify!($SelfT), ").next_multiple_of(-8), -16);")]
        #[doc = concat!("assert_eq!((-23_", stringify!($SelfT), ").next_multiple_of(-8), -24);")]
        /// ```
        #[unstable(feature = "int_roundings", issue = "88581")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[rustc_inherit_overflow_checks]
        pub const fn next_multiple_of(self, rhs: Self) -> Self {
            // This would otherwise fail when calculating `r` when self == T::MIN.
            if rhs == -1 {
                return self;
            }

            let r = self % rhs;
            let m = if (r > 0 && rhs < 0) || (r < 0 && rhs > 0) {
                r + rhs
            } else {
                r
            };

            if m == 0 {
                self
            } else {
                self + (rhs - m)
            }
        }

        /// If `rhs` is positive, calculates the smallest value greater than or
        /// equal to `self` that is a multiple of `rhs`. If `rhs` is negative,
        /// calculates the largest value less than or equal to `self` that is a
        /// multiple of `rhs`. Returns `None` if `rhs` is zero or the operation
        /// would result in overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(int_roundings)]
        #[doc = concat!("assert_eq!(16_", stringify!($SelfT), ".checked_next_multiple_of(8), Some(16));")]
        #[doc = concat!("assert_eq!(23_", stringify!($SelfT), ".checked_next_multiple_of(8), Some(24));")]
        #[doc = concat!("assert_eq!(16_", stringify!($SelfT), ".checked_next_multiple_of(-8), Some(16));")]
        #[doc = concat!("assert_eq!(23_", stringify!($SelfT), ".checked_next_multiple_of(-8), Some(16));")]
        #[doc = concat!("assert_eq!((-16_", stringify!($SelfT), ").checked_next_multiple_of(8), Some(-16));")]
        #[doc = concat!("assert_eq!((-23_", stringify!($SelfT), ").checked_next_multiple_of(8), Some(-16));")]
        #[doc = concat!("assert_eq!((-16_", stringify!($SelfT), ").checked_next_multiple_of(-8), Some(-16));")]
        #[doc = concat!("assert_eq!((-23_", stringify!($SelfT), ").checked_next_multiple_of(-8), Some(-24));")]
        #[doc = concat!("assert_eq!(1_", stringify!($SelfT), ".checked_next_multiple_of(0), None);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.checked_next_multiple_of(2), None);")]
        /// ```
        #[unstable(feature = "int_roundings", issue = "88581")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_next_multiple_of(self, rhs: Self) -> Option<Self> {
            // This would otherwise fail when calculating `r` when self == T::MIN.
            if rhs == -1 {
                return Some(self);
            }

            let r = try_opt!(self.checked_rem(rhs));
            let m = if (r > 0 && rhs < 0) || (r < 0 && rhs > 0) {
                // r + rhs cannot overflow because they have opposite signs
                r + rhs
            } else {
                r
            };

            if m == 0 {
                Some(self)
            } else {
                // rhs - m cannot overflow because m has the same sign as rhs
                self.checked_add(rhs - m)
            }
        }

        /// Returns the logarithm of the number with respect to an arbitrary base,
        /// rounded down.
        ///
        /// This method might not be optimized owing to implementation details;
        /// `ilog2` can produce results more efficiently for base 2, and `ilog10`
        /// can produce results more efficiently for base 10.
        ///
        /// # Panics
        ///
        /// This function will panic if `self` is less than or equal to zero,
        /// or if `base` is less than 2.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".ilog(5), 1);")]
        /// ```
        #[stable(feature = "int_log", since = "1.67.0")]
        #[rustc_const_stable(feature = "int_log", since = "1.67.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn ilog(self, base: Self) -> u32 {
            assert!(base >= 2, "base of integer logarithm must be at least 2");
            if let Some(log) = self.checked_ilog(base) {
                log
            } else {
                int_log10::panic_for_nonpositive_argument()
            }
        }

        /// Returns the base 2 logarithm of the number, rounded down.
        ///
        /// # Panics
        ///
        /// This function will panic if `self` is less than or equal to zero.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("assert_eq!(2", stringify!($SelfT), ".ilog2(), 1);")]
        /// ```
        #[stable(feature = "int_log", since = "1.67.0")]
        #[rustc_const_stable(feature = "int_log", since = "1.67.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn ilog2(self) -> u32 {
            if let Some(log) = self.checked_ilog2() {
                log
            } else {
                int_log10::panic_for_nonpositive_argument()
            }
        }

        /// Returns the base 10 logarithm of the number, rounded down.
        ///
        /// # Panics
        ///
        /// This function will panic if `self` is less than or equal to zero.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("assert_eq!(10", stringify!($SelfT), ".ilog10(), 1);")]
        /// ```
        #[stable(feature = "int_log", since = "1.67.0")]
        #[rustc_const_stable(feature = "int_log", since = "1.67.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn ilog10(self) -> u32 {
            if let Some(log) = self.checked_ilog10() {
                log
            } else {
                int_log10::panic_for_nonpositive_argument()
            }
        }

        /// Returns the logarithm of the number with respect to an arbitrary base,
        /// rounded down.
        ///
        /// Returns `None` if the number is negative or zero, or if the base is not at least 2.
        ///
        /// This method might not be optimized owing to implementation details;
        /// `checked_ilog2` can produce results more efficiently for base 2, and
        /// `checked_ilog10` can produce results more efficiently for base 10.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".checked_ilog(5), Some(1));")]
        /// ```
        #[stable(feature = "int_log", since = "1.67.0")]
        #[rustc_const_stable(feature = "int_log", since = "1.67.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_ilog(self, base: Self) -> Option<u32> {
            if self <= 0 || base <= 1 {
                None
            } else {
                // Delegate to the unsigned implementation.
                // The condition makes sure that both casts are exact.
                (self as $UnsignedT).checked_ilog(base as $UnsignedT)
            }
        }

        /// Returns the base 2 logarithm of the number, rounded down.
        ///
        /// Returns `None` if the number is negative or zero.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("assert_eq!(2", stringify!($SelfT), ".checked_ilog2(), Some(1));")]
        /// ```
        #[stable(feature = "int_log", since = "1.67.0")]
        #[rustc_const_stable(feature = "int_log", since = "1.67.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_ilog2(self) -> Option<u32> {
            if self <= 0 {
                None
            } else {
                // SAFETY: We just checked that this number is positive
                let log = (Self::BITS - 1) - unsafe { intrinsics::ctlz_nonzero(self) as u32 };
                Some(log)
            }
        }

        /// Returns the base 10 logarithm of the number, rounded down.
        ///
        /// Returns `None` if the number is negative or zero.
        ///
        /// # Example
        ///
        /// ```
        #[doc = concat!("assert_eq!(10", stringify!($SelfT), ".checked_ilog10(), Some(1));")]
        /// ```
        #[stable(feature = "int_log", since = "1.67.0")]
        #[rustc_const_stable(feature = "int_log", since = "1.67.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_ilog10(self) -> Option<u32> {
            if self > 0 {
                Some(int_log10::$ActualT(self as $ActualT))
            } else {
                None
            }
        }

        /// Computes the absolute value of `self`.
        ///
        /// # Overflow behavior
        ///
        /// The absolute value of
        #[doc = concat!("`", stringify!($SelfT), "::MIN`")]
        /// cannot be represented as an
        #[doc = concat!("`", stringify!($SelfT), "`,")]
        /// and attempting to calculate it will cause an overflow. This means
        /// that code in debug mode will trigger a panic on this case and
        /// optimized code will return
        #[doc = concat!("`", stringify!($SelfT), "::MIN`")]
        /// without a panic. If you do not want this behavior, consider
        /// using [`unsigned_abs`](Self::unsigned_abs) instead.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(10", stringify!($SelfT), ".abs(), 10);")]
        #[doc = concat!("assert_eq!((-10", stringify!($SelfT), ").abs(), 10);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[allow(unused_attributes)]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[rustc_inherit_overflow_checks]
        pub const fn abs(self) -> Self {
            // Note that the #[rustc_inherit_overflow_checks] and #[inline]
            // above mean that the overflow semantics of the subtraction
            // depend on the crate we're being called from.
            if self.is_negative() {
                -self
            } else {
                self
            }
        }

        /// Computes the absolute difference between `self` and `other`.
        ///
        /// This function always returns the correct answer without overflow or
        /// panics by returning an unsigned integer.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".abs_diff(80), 20", stringify!($UnsignedT), ");")]
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".abs_diff(110), 10", stringify!($UnsignedT), ");")]
        #[doc = concat!("assert_eq!((-100", stringify!($SelfT), ").abs_diff(80), 180", stringify!($UnsignedT), ");")]
        #[doc = concat!("assert_eq!((-100", stringify!($SelfT), ").abs_diff(-120), 20", stringify!($UnsignedT), ");")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN.abs_diff(", stringify!($SelfT), "::MAX), ", stringify!($UnsignedT), "::MAX);")]
        /// ```
        #[stable(feature = "int_abs_diff", since = "1.60.0")]
        #[rustc_const_stable(feature = "int_abs_diff", since = "1.60.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn abs_diff(self, other: Self) -> $UnsignedT {
            if self < other {
                // Converting a non-negative x from signed to unsigned by using
                // `x as U` is left unchanged, but a negative x is converted
                // to value x + 2^N. Thus if `s` and `o` are binary variables
                // respectively indicating whether `self` and `other` are
                // negative, we are computing the mathematical value:
                //
                //    (other + o*2^N) - (self + s*2^N)    mod  2^N
                //    other - self + (o-s)*2^N            mod  2^N
                //    other - self                        mod  2^N
                //
                // Finally, taking the mod 2^N of the mathematical value of
                // `other - self` does not change it as it already is
                // in the range [0, 2^N).
                (other as $UnsignedT).wrapping_sub(self as $UnsignedT)
            } else {
                (self as $UnsignedT).wrapping_sub(other as $UnsignedT)
            }
        }

        /// Returns a number representing sign of `self`.
        ///
        ///  - `0` if the number is zero
        ///  - `1` if the number is positive
        ///  - `-1` if the number is negative
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(10", stringify!($SelfT), ".signum(), 1);")]
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".signum(), 0);")]
        #[doc = concat!("assert_eq!((-10", stringify!($SelfT), ").signum(), -1);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_sign", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn signum(self) -> Self {
            // Picking the right way to phrase this is complicated
            // (<https://graphics.stanford.edu/~seander/bithacks.html#CopyIntegerSign>)
            // so delegate it to `Ord` which is already producing -1/0/+1
            // exactly like we need and can be the place to deal with the complexity.

            crate::intrinsics::three_way_compare(self, 0) as Self
        }

        /// Returns `true` if `self` is positive and `false` if the number is zero or
        /// negative.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert!(10", stringify!($SelfT), ".is_positive());")]
        #[doc = concat!("assert!(!(-10", stringify!($SelfT), ").is_positive());")]
        /// ```
        #[must_use]
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[inline(always)]
        pub const fn is_positive(self) -> bool { self > 0 }

        /// Returns `true` if `self` is negative and `false` if the number is zero or
        /// positive.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert!((-10", stringify!($SelfT), ").is_negative());")]
        #[doc = concat!("assert!(!10", stringify!($SelfT), ".is_negative());")]
        /// ```
        #[must_use]
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_methods", since = "1.32.0")]
        #[inline(always)]
        pub const fn is_negative(self) -> bool { self < 0 }

        /// Returns the memory representation of this integer as a byte array in
        /// big-endian (network) byte order.
        ///
        #[doc = $to_xe_bytes_doc]
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("let bytes = ", $swap_op, stringify!($SelfT), ".to_be_bytes();")]
        #[doc = concat!("assert_eq!(bytes, ", $be_bytes, ");")]
        /// ```
        #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
        #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn to_be_bytes(self) -> [u8; size_of::<Self>()] {
            self.to_be().to_ne_bytes()
        }

        /// Returns the memory representation of this integer as a byte array in
        /// little-endian byte order.
        ///
        #[doc = $to_xe_bytes_doc]
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("let bytes = ", $swap_op, stringify!($SelfT), ".to_le_bytes();")]
        #[doc = concat!("assert_eq!(bytes, ", $le_bytes, ");")]
        /// ```
        #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
        #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn to_le_bytes(self) -> [u8; size_of::<Self>()] {
            self.to_le().to_ne_bytes()
        }

        /// Returns the memory representation of this integer as a byte array in
        /// native byte order.
        ///
        /// As the target platform's native endianness is used, portable code
        /// should use [`to_be_bytes`] or [`to_le_bytes`], as appropriate,
        /// instead.
        ///
        #[doc = $to_xe_bytes_doc]
        ///
        /// [`to_be_bytes`]: Self::to_be_bytes
        /// [`to_le_bytes`]: Self::to_le_bytes
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("let bytes = ", $swap_op, stringify!($SelfT), ".to_ne_bytes();")]
        /// assert_eq!(
        ///     bytes,
        ///     if cfg!(target_endian = "big") {
        #[doc = concat!("        ", $be_bytes)]
        ///     } else {
        #[doc = concat!("        ", $le_bytes)]
        ///     }
        /// );
        /// ```
        #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
        #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
        // SAFETY: const sound because integers are plain old datatypes so we can always
        // transmute them to arrays of bytes
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn to_ne_bytes(self) -> [u8; size_of::<Self>()] {
            // SAFETY: integers are plain old datatypes so we can always transmute them to
            // arrays of bytes
            unsafe { mem::transmute(self) }
        }

        /// Creates an integer value from its representation as a byte array in
        /// big endian.
        ///
        #[doc = $from_xe_bytes_doc]
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("let value = ", stringify!($SelfT), "::from_be_bytes(", $be_bytes, ");")]
        #[doc = concat!("assert_eq!(value, ", $swap_op, ");")]
        /// ```
        ///
        /// When starting from a slice rather than an array, fallible conversion APIs can be used:
        ///
        /// ```
        #[doc = concat!("fn read_be_", stringify!($SelfT), "(input: &mut &[u8]) -> ", stringify!($SelfT), " {")]
        #[doc = concat!("    let (int_bytes, rest) = input.split_at(size_of::<", stringify!($SelfT), ">());")]
        ///     *input = rest;
        #[doc = concat!("    ", stringify!($SelfT), "::from_be_bytes(int_bytes.try_into().unwrap())")]
        /// }
        /// ```
        #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
        #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
        #[must_use]
        #[inline]
        pub const fn from_be_bytes(bytes: [u8; size_of::<Self>()]) -> Self {
            Self::from_be(Self::from_ne_bytes(bytes))
        }

        /// Creates an integer value from its representation as a byte array in
        /// little endian.
        ///
        #[doc = $from_xe_bytes_doc]
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("let value = ", stringify!($SelfT), "::from_le_bytes(", $le_bytes, ");")]
        #[doc = concat!("assert_eq!(value, ", $swap_op, ");")]
        /// ```
        ///
        /// When starting from a slice rather than an array, fallible conversion APIs can be used:
        ///
        /// ```
        #[doc = concat!("fn read_le_", stringify!($SelfT), "(input: &mut &[u8]) -> ", stringify!($SelfT), " {")]
        #[doc = concat!("    let (int_bytes, rest) = input.split_at(size_of::<", stringify!($SelfT), ">());")]
        ///     *input = rest;
        #[doc = concat!("    ", stringify!($SelfT), "::from_le_bytes(int_bytes.try_into().unwrap())")]
        /// }
        /// ```
        #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
        #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
        #[must_use]
        #[inline]
        pub const fn from_le_bytes(bytes: [u8; size_of::<Self>()]) -> Self {
            Self::from_le(Self::from_ne_bytes(bytes))
        }

        /// Creates an integer value from its memory representation as a byte
        /// array in native endianness.
        ///
        /// As the target platform's native endianness is used, portable code
        /// likely wants to use [`from_be_bytes`] or [`from_le_bytes`], as
        /// appropriate instead.
        ///
        /// [`from_be_bytes`]: Self::from_be_bytes
        /// [`from_le_bytes`]: Self::from_le_bytes
        ///
        #[doc = $from_xe_bytes_doc]
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("let value = ", stringify!($SelfT), "::from_ne_bytes(if cfg!(target_endian = \"big\") {")]
        #[doc = concat!("    ", $be_bytes)]
        /// } else {
        #[doc = concat!("    ", $le_bytes)]
        /// });
        #[doc = concat!("assert_eq!(value, ", $swap_op, ");")]
        /// ```
        ///
        /// When starting from a slice rather than an array, fallible conversion APIs can be used:
        ///
        /// ```
        #[doc = concat!("fn read_ne_", stringify!($SelfT), "(input: &mut &[u8]) -> ", stringify!($SelfT), " {")]
        #[doc = concat!("    let (int_bytes, rest) = input.split_at(size_of::<", stringify!($SelfT), ">());")]
        ///     *input = rest;
        #[doc = concat!("    ", stringify!($SelfT), "::from_ne_bytes(int_bytes.try_into().unwrap())")]
        /// }
        /// ```
        #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
        #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
        #[must_use]
        // SAFETY: const sound because integers are plain old datatypes so we can always
        // transmute to them
        #[inline]
        pub const fn from_ne_bytes(bytes: [u8; size_of::<Self>()]) -> Self {
            // SAFETY: integers are plain old datatypes so we can always transmute to them
            unsafe { mem::transmute(bytes) }
        }

        /// New code should prefer to use
        #[doc = concat!("[`", stringify!($SelfT), "::MIN", "`] instead.")]
        ///
        /// Returns the smallest value that can be represented by this integer type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline(always)]
        #[rustc_promotable]
        #[rustc_const_stable(feature = "const_min_value", since = "1.32.0")]
        #[deprecated(since = "TBD", note = "replaced by the `MIN` associated constant on this type")]
        #[rustc_diagnostic_item = concat!(stringify!($SelfT), "_legacy_fn_min_value")]
        pub const fn min_value() -> Self {
            Self::MIN
        }

        /// New code should prefer to use
        #[doc = concat!("[`", stringify!($SelfT), "::MAX", "`] instead.")]
        ///
        /// Returns the largest value that can be represented by this integer type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[inline(always)]
        #[rustc_promotable]
        #[rustc_const_stable(feature = "const_max_value", since = "1.32.0")]
        #[deprecated(since = "TBD", note = "replaced by the `MAX` associated constant on this type")]
        #[rustc_diagnostic_item = concat!(stringify!($SelfT), "_legacy_fn_max_value")]
        pub const fn max_value() -> Self {
            Self::MAX
        }
    }
}
