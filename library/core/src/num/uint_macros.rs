macro_rules! uint_impl {
    (
        Self = $SelfT:ty,
        ActualT = $ActualT:ident,
        SignedT = $SignedT:ident,

        // These are all for use *only* in doc comments.
        // As such, they're all passed as literals -- passing them as a string
        // literal is fine if they need to be multiple code tokens.
        // In non-comments, use the associated constants rather than these.
        BITS = $BITS:literal,
        BITS_MINUS_ONE = $BITS_MINUS_ONE:literal,
        MAX = $MaxV:literal,
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
        /// The smallest value that can be represented by this integer type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MIN, 0);")]
        /// ```
        #[stable(feature = "assoc_int_consts", since = "1.43.0")]
        pub const MIN: Self = 0;

        /// The largest value that can be represented by this integer type
        #[doc = concat!("(2<sup>", $BITS, "</sup> &minus; 1", $bound_condition, ").")]
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX, ", stringify!($MaxV), ");")]
        /// ```
        #[stable(feature = "assoc_int_consts", since = "1.43.0")]
        pub const MAX: Self = !0;

        /// The size of this integer type in bits.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::BITS, ", stringify!($BITS), ");")]
        /// ```
        #[stable(feature = "int_bits_const", since = "1.53.0")]
        pub const BITS: u32 = Self::MAX.count_ones();

        /// Returns the number of ones in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = 0b01001100", stringify!($SelfT), ";")]
        /// assert_eq!(n.count_ones(), 3);
        ///
        #[doc = concat!("let max = ", stringify!($SelfT),"::MAX;")]
        #[doc = concat!("assert_eq!(max.count_ones(), ", stringify!($BITS), ");")]
        ///
        #[doc = concat!("let zero = 0", stringify!($SelfT), ";")]
        /// assert_eq!(zero.count_ones(), 0);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
        #[doc(alias = "popcount")]
        #[doc(alias = "popcnt")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn count_ones(self) -> u32 {
            return intrinsics::ctpop(self);
        }

        /// Returns the number of zeros in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let zero = 0", stringify!($SelfT), ";")]
        #[doc = concat!("assert_eq!(zero.count_zeros(), ", stringify!($BITS), ");")]
        ///
        #[doc = concat!("let max = ", stringify!($SelfT),"::MAX;")]
        /// assert_eq!(max.count_zeros(), 0);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
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
        #[doc = concat!("let n = ", stringify!($SelfT), "::MAX >> 2;")]
        /// assert_eq!(n.leading_zeros(), 2);
        ///
        #[doc = concat!("let zero = 0", stringify!($SelfT), ";")]
        #[doc = concat!("assert_eq!(zero.leading_zeros(), ", stringify!($BITS), ");")]
        ///
        #[doc = concat!("let max = ", stringify!($SelfT),"::MAX;")]
        /// assert_eq!(max.leading_zeros(), 0);
        /// ```
        #[doc = concat!("[`ilog2`]: ", stringify!($SelfT), "::ilog2")]
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn leading_zeros(self) -> u32 {
            return intrinsics::ctlz(self as $ActualT);
        }

        /// Returns the number of trailing zeros in the binary representation
        /// of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = 0b0101000", stringify!($SelfT), ";")]
        /// assert_eq!(n.trailing_zeros(), 3);
        ///
        #[doc = concat!("let zero = 0", stringify!($SelfT), ";")]
        #[doc = concat!("assert_eq!(zero.trailing_zeros(), ", stringify!($BITS), ");")]
        ///
        #[doc = concat!("let max = ", stringify!($SelfT),"::MAX;")]
        #[doc = concat!("assert_eq!(max.trailing_zeros(), 0);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn trailing_zeros(self) -> u32 {
            return intrinsics::cttz(self);
        }

        /// Returns the number of leading ones in the binary representation of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = !(", stringify!($SelfT), "::MAX >> 2);")]
        /// assert_eq!(n.leading_ones(), 2);
        ///
        #[doc = concat!("let zero = 0", stringify!($SelfT), ";")]
        /// assert_eq!(zero.leading_ones(), 0);
        ///
        #[doc = concat!("let max = ", stringify!($SelfT),"::MAX;")]
        #[doc = concat!("assert_eq!(max.leading_ones(), ", stringify!($BITS), ");")]
        /// ```
        #[stable(feature = "leading_trailing_ones", since = "1.46.0")]
        #[rustc_const_stable(feature = "leading_trailing_ones", since = "1.46.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn leading_ones(self) -> u32 {
            (!self).leading_zeros()
        }

        /// Returns the number of trailing ones in the binary representation
        /// of `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = 0b1010111", stringify!($SelfT), ";")]
        /// assert_eq!(n.trailing_ones(), 3);
        ///
        #[doc = concat!("let zero = 0", stringify!($SelfT), ";")]
        /// assert_eq!(zero.trailing_ones(), 0);
        ///
        #[doc = concat!("let max = ", stringify!($SelfT),"::MAX;")]
        #[doc = concat!("assert_eq!(max.trailing_ones(), ", stringify!($BITS), ");")]
        /// ```
        #[stable(feature = "leading_trailing_ones", since = "1.46.0")]
        #[rustc_const_stable(feature = "leading_trailing_ones", since = "1.46.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn trailing_ones(self) -> u32 {
            (!self).trailing_zeros()
        }

        /// Returns the bit pattern of `self` reinterpreted as a signed integer of the same size.
        ///
        /// This produces the same result as an `as` cast, but ensures that the bit-width remains
        /// the same.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(integer_sign_cast)]
        ///
        #[doc = concat!("let n = ", stringify!($SelfT), "::MAX;")]
        ///
        #[doc = concat!("assert_eq!(n.cast_signed(), -1", stringify!($SignedT), ");")]
        /// ```
        #[unstable(feature = "integer_sign_cast", issue = "125882")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn cast_signed(self) -> $SignedT {
            self as $SignedT
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
        #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn rotate_left(self, n: u32) -> Self {
            return intrinsics::rotate_left(self, n);
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
        #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn rotate_right(self, n: u32) -> Self {
            return intrinsics::rotate_right(self, n);
        }

        /// Reverses the byte order of the integer.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("let n = ", $swap_op, stringify!($SelfT), ";")]
        /// let m = n.swap_bytes();
        ///
        #[doc = concat!("assert_eq!(m, ", $swapped, ");")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn swap_bytes(self) -> Self {
            intrinsics::bswap(self as $ActualT) as Self
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
            intrinsics::bitreverse(self as $ActualT) as Self
        }

        /// Converts an integer from big endian to the target's endianness.
        ///
        /// On big endian this is a no-op. On little endian the bytes are
        /// swapped.
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
        #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
        #[must_use]
        #[inline(always)]
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
        /// On little endian this is a no-op. On big endian the bytes are
        /// swapped.
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
        #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
        #[must_use]
        #[inline(always)]
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
        /// On big endian this is a no-op. On little endian the bytes are
        /// swapped.
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
        #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
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
        /// On little endian this is a no-op. On big endian the bytes are
        /// swapped.
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
        #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
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
        #[doc = concat!(
            "assert_eq!((", stringify!($SelfT), "::MAX - 2).checked_add(1), ",
            "Some(", stringify!($SelfT), "::MAX - 1));"
        )]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).checked_add(3), None);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_add(self, rhs: Self) -> Option<Self> {
            // This used to use `overflowing_add`, but that means it ends up being
            // a `wrapping_add`, losing some optimization opportunities. Notably,
            // phrasing it this way helps `.checked_add(1)` optimize to a check
            // against `MAX` and a `add nuw`.
            // Per <https://github.com/rust-lang/rust/pull/124114#issuecomment-2066173305>,
            // LLVM is happy to re-form the intrinsic later if useful.

            if intrinsics::unlikely(intrinsics::add_with_overflow(self, rhs).1) {
                None
            } else {
                // SAFETY: Just checked it doesn't overflow
                Some(unsafe { intrinsics::unchecked_add(self, rhs) })
            }
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

        /// Checked addition with a signed integer. Computes `self + rhs`,
        /// returning `None` if overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".checked_add_signed(2), Some(3));")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".checked_add_signed(-2), None);")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).checked_add_signed(3), None);")]
        /// ```
        #[stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[rustc_const_stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_add_signed(self, rhs: $SignedT) -> Option<Self> {
            let (a, b) = self.overflowing_add_signed(rhs);
            if intrinsics::unlikely(b) { None } else { Some(a) }
        }

        /// Strict addition with a signed integer. Computes `self + rhs`,
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
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".strict_add_signed(2), 3);")]
        /// ```
        ///
        /// The following panic because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = 1", stringify!($SelfT), ".strict_add_signed(-2);")]
        /// ```
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = (", stringify!($SelfT), "::MAX - 2).strict_add_signed(3);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn strict_add_signed(self, rhs: $SignedT) -> Self {
            let (a, b) = self.overflowing_add_signed(rhs);
            if b { overflow_panic::add() } else { a }
         }

        /// Checked integer subtraction. Computes `self - rhs`, returning
        /// `None` if overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".checked_sub(1), Some(0));")]
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".checked_sub(1), None);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
            // Per PR#103299, there's no advantage to the `overflowing` intrinsic
            // for *unsigned* subtraction and we just emit the manual check anyway.
            // Thus, rather than using `overflowing_sub` that produces a wrapping
            // subtraction, check it ourself so we can use an unchecked one.

            if self < rhs {
                None
            } else {
                // SAFETY: just checked this can't overflow
                Some(unsafe { intrinsics::unchecked_sub(self, rhs) })
            }
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
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".strict_sub(1), 0);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = 0", stringify!($SelfT), ".strict_sub(1);")]
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
        /// If you find yourself writing code like this:
        ///
        /// ```
        /// # let foo = 30_u32;
        /// # let bar = 20;
        /// if foo >= bar {
        ///     // SAFETY: just checked it will not overflow
        ///     let diff = unsafe { foo.unchecked_sub(bar) };
        ///     // ... use diff ...
        /// }
        /// ```
        ///
        /// Consider changing it to
        ///
        /// ```
        /// # let foo = 30_u32;
        /// # let bar = 20;
        /// if let Some(diff) = foo.checked_sub(bar) {
        ///     // ... use diff ...
        /// }
        /// ```
        ///
        /// As that does exactly the same thing -- including telling the optimizer
        /// that the subtraction cannot overflow -- but avoids needing `unsafe`.
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

        /// Checked subtraction with a signed integer. Computes `self - rhs`,
        /// returning `None` if overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(mixed_integer_ops_unsigned_sub)]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".checked_sub_signed(2), None);")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".checked_sub_signed(-2), Some(3));")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).checked_sub_signed(-4), None);")]
        /// ```
        #[unstable(feature = "mixed_integer_ops_unsigned_sub", issue = "126043")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_sub_signed(self, rhs: $SignedT) -> Option<Self> {
            let (res, overflow) = self.overflowing_sub_signed(rhs);

            if !overflow {
                Some(res)
            } else {
                None
            }
        }

        #[doc = concat!(
            "Checked integer subtraction. Computes `self - rhs` and checks if the result fits into an [`",
            stringify!($SignedT), "`], returning `None` if overflow occurred."
        )]
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(unsigned_signed_diff)]
        #[doc = concat!("assert_eq!(10", stringify!($SelfT), ".checked_signed_diff(2), Some(8));")]
        #[doc = concat!("assert_eq!(2", stringify!($SelfT), ".checked_signed_diff(10), Some(-8));")]
        #[doc = concat!(
            "assert_eq!(",
            stringify!($SelfT),
            "::MAX.checked_signed_diff(",
            stringify!($SignedT),
            "::MAX as ",
            stringify!($SelfT),
            "), None);"
        )]
        #[doc = concat!(
            "assert_eq!((",
            stringify!($SignedT),
            "::MAX as ",
            stringify!($SelfT),
            ").checked_signed_diff(",
            stringify!($SelfT),
            "::MAX), Some(",
            stringify!($SignedT),
            "::MIN));"
        )]
        #[doc = concat!(
            "assert_eq!((",
            stringify!($SignedT),
            "::MAX as ",
            stringify!($SelfT),
            " + 1).checked_signed_diff(0), None);"
        )]
        #[doc = concat!(
            "assert_eq!(",
            stringify!($SelfT),
            "::MAX.checked_signed_diff(",
            stringify!($SelfT),
            "::MAX), Some(0));"
        )]
        /// ```
        #[unstable(feature = "unsigned_signed_diff", issue = "126041")]
        #[inline]
        pub const fn checked_signed_diff(self, rhs: Self) -> Option<$SignedT> {
            let res = self.wrapping_sub(rhs) as $SignedT;
            let overflow = (self >= rhs) == (res < 0);

            if !overflow {
                Some(res)
            } else {
                None
            }
        }

        /// Checked integer multiplication. Computes `self * rhs`, returning
        /// `None` if overflow occurred.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".checked_mul(1), Some(5));")]
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
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".strict_mul(1), 5);")]
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

        /// Checked integer division. Computes `self / rhs`, returning `None`
        /// if `rhs == 0`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(128", stringify!($SelfT), ".checked_div(2), Some(64));")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".checked_div(0), None);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_checked_int_div", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_div(self, rhs: Self) -> Option<Self> {
            if intrinsics::unlikely(rhs == 0) {
                None
            } else {
                // SAFETY: div by zero has been checked above and unsigned types have no other
                // failure modes for division
                Some(unsafe { intrinsics::unchecked_div(self, rhs) })
            }
        }

        /// Strict integer division. Computes `self / rhs`.
        ///
        /// Strict division on unsigned types is just normal division. There's no
        /// way overflow could ever happen. This function exists so that all
        /// operations are accounted for in the strict operations.
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
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".strict_div(10), 10);")]
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
        #[inline(always)]
        #[track_caller]
        pub const fn strict_div(self, rhs: Self) -> Self {
            self / rhs
        }

        /// Checked Euclidean division. Computes `self.div_euclid(rhs)`, returning `None`
        /// if `rhs == 0`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(128", stringify!($SelfT), ".checked_div_euclid(2), Some(64));")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".checked_div_euclid(0), None);")]
        /// ```
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_div_euclid(self, rhs: Self) -> Option<Self> {
            if intrinsics::unlikely(rhs == 0) {
                None
            } else {
                Some(self.div_euclid(rhs))
            }
        }

        /// Strict Euclidean division. Computes `self.div_euclid(rhs)`.
        ///
        /// Strict division on unsigned types is just normal division. There's no
        /// way overflow could ever happen. This function exists so that all
        /// operations are accounted for in the strict operations. Since, for the
        /// positive integers, all common definitions of division are equal, this
        /// is exactly equal to `self.strict_div(rhs)`.
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
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".strict_div_euclid(10), 10);")]
        /// ```
        /// The following panics because of division by zero:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = (1", stringify!($SelfT), ").strict_div_euclid(0);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[track_caller]
        pub const fn strict_div_euclid(self, rhs: Self) -> Self {
            self / rhs
        }

        /// Checked integer remainder. Computes `self % rhs`, returning `None`
        /// if `rhs == 0`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".checked_rem(2), Some(1));")]
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".checked_rem(0), None);")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_checked_int_div", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_rem(self, rhs: Self) -> Option<Self> {
            if intrinsics::unlikely(rhs == 0) {
                None
            } else {
                // SAFETY: div by zero has been checked above and unsigned types have no other
                // failure modes for division
                Some(unsafe { intrinsics::unchecked_rem(self, rhs) })
            }
        }

        /// Strict integer remainder. Computes `self % rhs`.
        ///
        /// Strict remainder calculation on unsigned types is just the regular
        /// remainder calculation. There's no way overflow could ever happen.
        /// This function exists so that all operations are accounted for in the
        /// strict operations.
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
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".strict_rem(10), 0);")]
        /// ```
        ///
        /// The following panics because of division by zero:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = 5", stringify!($SelfT), ".strict_rem(0);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[track_caller]
        pub const fn strict_rem(self, rhs: Self) -> Self {
            self % rhs
        }

        /// Checked Euclidean modulo. Computes `self.rem_euclid(rhs)`, returning `None`
        /// if `rhs == 0`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".checked_rem_euclid(2), Some(1));")]
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".checked_rem_euclid(0), None);")]
        /// ```
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_rem_euclid(self, rhs: Self) -> Option<Self> {
            if intrinsics::unlikely(rhs == 0) {
                None
            } else {
                Some(self.rem_euclid(rhs))
            }
        }

        /// Strict Euclidean modulo. Computes `self.rem_euclid(rhs)`.
        ///
        /// Strict modulo calculation on unsigned types is just the regular
        /// remainder calculation. There's no way overflow could ever happen.
        /// This function exists so that all operations are accounted for in the
        /// strict operations. Since, for the positive integers, all common
        /// definitions of division are equal, this is exactly equal to
        /// `self.strict_rem(rhs)`.
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
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".strict_rem_euclid(10), 0);")]
        /// ```
        ///
        /// The following panics because of division by zero:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = 5", stringify!($SelfT), ".strict_rem_euclid(0);")]
        /// ```
        #[unstable(feature = "strict_overflow_ops", issue = "118260")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[track_caller]
        pub const fn strict_rem_euclid(self, rhs: Self) -> Self {
            self % rhs
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
        /// This function will panic if `self` is zero, or if `base` is less than 2.
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
        /// This function will panic if `self` is zero.
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
        /// This function will panic if `self` is zero.
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
        /// Returns `None` if the number is zero, or if the base is not at least 2.
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
            } else if self < base {
                Some(0)
            } else {
                // Since base >= self, n >= 1
                let mut n = 1;
                let mut r = base;

                // Optimization for 128 bit wide integers.
                if Self::BITS == 128 {
                    // The following is a correct lower bound for ⌊log(base,self)⌋ because
                    //
                    // log(base,self) = log(2,self) / log(2,base)
                    //                ≥ ⌊log(2,self)⌋ / (⌊log(2,base)⌋ + 1)
                    //
                    // hence
                    //
                    // ⌊log(base,self)⌋ ≥ ⌊ ⌊log(2,self)⌋ / (⌊log(2,base)⌋ + 1) ⌋ .
                    n = self.ilog2() / (base.ilog2() + 1);
                    r = base.pow(n);
                }

                while r <= self / base {
                    n += 1;
                    r *= base;
                }
                Some(n)
            }
        }

        /// Returns the base 2 logarithm of the number, rounded down.
        ///
        /// Returns `None` if the number is zero.
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
            match NonZero::new(self) {
                Some(x) => Some(x.ilog2()),
                None => None,
            }
        }

        /// Returns the base 10 logarithm of the number, rounded down.
        ///
        /// Returns `None` if the number is zero.
        ///
        /// # Examples
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
            match NonZero::new(self) {
                Some(x) => Some(x.ilog10()),
                None => None,
            }
        }

        /// Checked negation. Computes `-self`, returning `None` unless `self ==
        /// 0`.
        ///
        /// Note that negating any positive integer will overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".checked_neg(), Some(0));")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".checked_neg(), None);")]
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

        /// Strict negation. Computes `-self`, panicking unless `self ==
        /// 0`.
        ///
        /// Note that negating any positive integer will overflow.
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
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".strict_neg(), 0);")]
        /// ```
        ///
        /// The following panics because of overflow:
        ///
        /// ```should_panic
        /// #![feature(strict_overflow_ops)]
        #[doc = concat!("let _ = 1", stringify!($SelfT), ".strict_neg();")]
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

        /// Checked shift left. Computes `self << rhs`, returning `None`
        /// if `rhs` is larger than or equal to the number of bits in `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0x1", stringify!($SelfT), ".checked_shl(4), Some(0x10));")]
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".checked_shl(129), None);")]
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
        #[doc = concat!("let _ = 0x10", stringify!($SelfT), ".strict_shl(129);")]
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
        /// #![feature(unbounded_shifts)]
        #[doc = concat!("assert_eq!(0x1", stringify!($SelfT), ".unbounded_shl(4), 0x10);")]
        #[doc = concat!("assert_eq!(0x1", stringify!($SelfT), ".unbounded_shl(129), 0);")]
        /// ```
        #[unstable(feature = "unbounded_shifts", issue = "129375")]
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

        /// Checked shift right. Computes `self >> rhs`, returning `None`
        /// if `rhs` is larger than or equal to the number of bits in `self`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".checked_shr(4), Some(0x1));")]
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".checked_shr(129), None);")]
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
        #[doc = concat!("let _ = 0x10", stringify!($SelfT), ".strict_shr(129);")]
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
        /// the entire value is shifted out, and `0` is returned.
        ///
        /// # Examples
        ///
        /// Basic usage:
        /// ```
        /// #![feature(unbounded_shifts)]
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".unbounded_shr(4), 0x1);")]
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".unbounded_shr(129), 0);")]
        /// ```
        #[unstable(feature = "unbounded_shifts", issue = "129375")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn unbounded_shr(self, rhs: u32) -> $SelfT{
            if rhs < Self::BITS {
                // SAFETY:
                // rhs is just checked to be in-range above
                unsafe { self.unchecked_shr(rhs) }
            } else {
                0
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
        #[doc = concat!("assert_eq!(2", stringify!($SelfT), ".checked_pow(5), Some(32));")]
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
        #[doc = concat!("assert_eq!(2", stringify!($SelfT), ".strict_pow(5), 32);")]
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

        /// Saturating integer addition. Computes `self + rhs`, saturating at
        /// the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".saturating_add(1), 101);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.saturating_add(127), ", stringify!($SelfT), "::MAX);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[rustc_const_stable(feature = "const_saturating_int_methods", since = "1.47.0")]
        #[inline(always)]
        pub const fn saturating_add(self, rhs: Self) -> Self {
            intrinsics::saturating_add(self, rhs)
        }

        /// Saturating addition with a signed integer. Computes `self + rhs`,
        /// saturating at the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".saturating_add_signed(2), 3);")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".saturating_add_signed(-2), 0);")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).saturating_add_signed(4), ", stringify!($SelfT), "::MAX);")]
        /// ```
        #[stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[rustc_const_stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn saturating_add_signed(self, rhs: $SignedT) -> Self {
            let (res, overflow) = self.overflowing_add(rhs as Self);
            if overflow == (rhs < 0) {
                res
            } else if overflow {
                Self::MAX
            } else {
                0
            }
        }

        /// Saturating integer subtraction. Computes `self - rhs`, saturating
        /// at the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".saturating_sub(27), 73);")]
        #[doc = concat!("assert_eq!(13", stringify!($SelfT), ".saturating_sub(127), 0);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[rustc_const_stable(feature = "const_saturating_int_methods", since = "1.47.0")]
        #[inline(always)]
        pub const fn saturating_sub(self, rhs: Self) -> Self {
            intrinsics::saturating_sub(self, rhs)
        }

        /// Saturating integer subtraction. Computes `self` - `rhs`, saturating at
        /// the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(mixed_integer_ops_unsigned_sub)]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".saturating_sub_signed(2), 0);")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".saturating_sub_signed(-2), 3);")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).saturating_sub_signed(-4), ", stringify!($SelfT), "::MAX);")]
        /// ```
        #[unstable(feature = "mixed_integer_ops_unsigned_sub", issue = "126043")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn saturating_sub_signed(self, rhs: $SignedT) -> Self {
            let (res, overflow) = self.overflowing_sub_signed(rhs);

            if !overflow {
                res
            } else if rhs < 0 {
                Self::MAX
            } else {
                0
            }
        }

        /// Saturating integer multiplication. Computes `self * rhs`,
        /// saturating at the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(2", stringify!($SelfT), ".saturating_mul(10), 20);")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX).saturating_mul(10), ", stringify!($SelfT),"::MAX);")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_saturating_int_methods", since = "1.47.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn saturating_mul(self, rhs: Self) -> Self {
            match self.checked_mul(rhs) {
                Some(x) => x,
                None => Self::MAX,
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
        ///
        /// ```
        #[stable(feature = "saturating_div", since = "1.58.0")]
        #[rustc_const_stable(feature = "saturating_div", since = "1.58.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn saturating_div(self, rhs: Self) -> Self {
            // on unsigned types, there is no overflow in integer division
            self.wrapping_div(rhs)
        }

        /// Saturating integer exponentiation. Computes `self.pow(exp)`,
        /// saturating at the numeric bounds instead of overflowing.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(4", stringify!($SelfT), ".saturating_pow(3), 64);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.saturating_pow(2), ", stringify!($SelfT), "::MAX);")]
        /// ```
        #[stable(feature = "no_panic_pow", since = "1.34.0")]
        #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn saturating_pow(self, exp: u32) -> Self {
            match self.checked_pow(exp) {
                Some(x) => x,
                None => Self::MAX,
            }
        }

        /// Wrapping (modular) addition. Computes `self + rhs`,
        /// wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(200", stringify!($SelfT), ".wrapping_add(55), 255);")]
        #[doc = concat!("assert_eq!(200", stringify!($SelfT), ".wrapping_add(", stringify!($SelfT), "::MAX), 199);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn wrapping_add(self, rhs: Self) -> Self {
            intrinsics::wrapping_add(self, rhs)
        }

        /// Wrapping (modular) addition with a signed integer. Computes
        /// `self + rhs`, wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".wrapping_add_signed(2), 3);")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".wrapping_add_signed(-2), ", stringify!($SelfT), "::MAX);")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).wrapping_add_signed(4), 1);")]
        /// ```
        #[stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[rustc_const_stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn wrapping_add_signed(self, rhs: $SignedT) -> Self {
            self.wrapping_add(rhs as Self)
        }

        /// Wrapping (modular) subtraction. Computes `self - rhs`,
        /// wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".wrapping_sub(100), 0);")]
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".wrapping_sub(", stringify!($SelfT), "::MAX), 101);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn wrapping_sub(self, rhs: Self) -> Self {
            intrinsics::wrapping_sub(self, rhs)
        }

        /// Wrapping (modular) subtraction with a signed integer. Computes
        /// `self - rhs`, wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(mixed_integer_ops_unsigned_sub)]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".wrapping_sub_signed(2), ", stringify!($SelfT), "::MAX);")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".wrapping_sub_signed(-2), 3);")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).wrapping_sub_signed(-4), 1);")]
        /// ```
        #[unstable(feature = "mixed_integer_ops_unsigned_sub", issue = "126043")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn wrapping_sub_signed(self, rhs: $SignedT) -> Self {
            self.wrapping_sub(rhs as Self)
        }

        /// Wrapping (modular) multiplication. Computes `self *
        /// rhs`, wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// Please note that this example is shared between integer types.
        /// Which explains why `u8` is used here.
        ///
        /// ```
        /// assert_eq!(10u8.wrapping_mul(12), 120);
        /// assert_eq!(25u8.wrapping_mul(12), 44);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn wrapping_mul(self, rhs: Self) -> Self {
            intrinsics::wrapping_mul(self, rhs)
        }

        /// Wrapping (modular) division. Computes `self / rhs`.
        ///
        /// Wrapped division on unsigned types is just normal division. There's
        /// no way wrapping could ever happen. This function exists so that all
        /// operations are accounted for in the wrapping operations.
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
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[rustc_const_stable(feature = "const_wrapping_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[track_caller]
        pub const fn wrapping_div(self, rhs: Self) -> Self {
            self / rhs
        }

        /// Wrapping Euclidean division. Computes `self.div_euclid(rhs)`.
        ///
        /// Wrapped division on unsigned types is just normal division. There's
        /// no way wrapping could ever happen. This function exists so that all
        /// operations are accounted for in the wrapping operations. Since, for
        /// the positive integers, all common definitions of division are equal,
        /// this is exactly equal to `self.wrapping_div(rhs)`.
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
        /// ```
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[track_caller]
        pub const fn wrapping_div_euclid(self, rhs: Self) -> Self {
            self / rhs
        }

        /// Wrapping (modular) remainder. Computes `self % rhs`.
        ///
        /// Wrapped remainder calculation on unsigned types is just the regular
        /// remainder calculation. There's no way wrapping could ever happen.
        /// This function exists so that all operations are accounted for in the
        /// wrapping operations.
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
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[rustc_const_stable(feature = "const_wrapping_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[track_caller]
        pub const fn wrapping_rem(self, rhs: Self) -> Self {
            self % rhs
        }

        /// Wrapping Euclidean modulo. Computes `self.rem_euclid(rhs)`.
        ///
        /// Wrapped modulo calculation on unsigned types is just the regular
        /// remainder calculation. There's no way wrapping could ever happen.
        /// This function exists so that all operations are accounted for in the
        /// wrapping operations. Since, for the positive integers, all common
        /// definitions of division are equal, this is exactly equal to
        /// `self.wrapping_rem(rhs)`.
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
        /// ```
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[track_caller]
        pub const fn wrapping_rem_euclid(self, rhs: Self) -> Self {
            self % rhs
        }

        /// Wrapping (modular) negation. Computes `-self`,
        /// wrapping around at the boundary of the type.
        ///
        /// Since unsigned types do not have negative equivalents
        /// all applications of this function will wrap (except for `-0`).
        /// For values smaller than the corresponding signed type's maximum
        /// the result is the same as casting the corresponding signed value.
        /// Any larger values are equivalent to `MAX + 1 - (val - MAX - 1)` where
        /// `MAX` is the corresponding signed type's maximum.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0_", stringify!($SelfT), ".wrapping_neg(), 0);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.wrapping_neg(), 1);")]
        #[doc = concat!("assert_eq!(13_", stringify!($SelfT), ".wrapping_neg(), (!13) + 1);")]
        #[doc = concat!("assert_eq!(42_", stringify!($SelfT), ".wrapping_neg(), !(42 - 1));")]
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn wrapping_neg(self) -> Self {
            (0 as $SelfT).wrapping_sub(self)
        }

        /// Panic-free bitwise shift-left; yields `self << mask(rhs)`,
        /// where `mask` removes any high-order bits of `rhs` that
        /// would cause the shift to exceed the bitwidth of the type.
        ///
        /// Note that this is *not* the same as a rotate-left; the
        /// RHS of a wrapping shift-left is restricted to the range
        /// of the type, rather than the bits shifted out of the LHS
        /// being returned to the other end. The primitive integer
        /// types all implement a [`rotate_left`](Self::rotate_left) function,
        /// which may be what you want instead.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".wrapping_shl(7), 128);")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".wrapping_shl(128), 1);")]
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
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
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(128", stringify!($SelfT), ".wrapping_shr(7), 1);")]
        #[doc = concat!("assert_eq!(128", stringify!($SelfT), ".wrapping_shr(128), 128);")]
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
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

        /// Wrapping (modular) exponentiation. Computes `self.pow(exp)`,
        /// wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(3", stringify!($SelfT), ".wrapping_pow(5), 243);")]
        /// assert_eq!(3u8.wrapping_pow(6), 217);
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
        /// whether an arithmetic overflow would occur. If an overflow would
        /// have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".overflowing_add(2), (7, false));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.overflowing_add(1), (0, true));")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn overflowing_add(self, rhs: Self) -> (Self, bool) {
            let (a, b) = intrinsics::add_with_overflow(self as $ActualT, rhs as $ActualT);
            (a as Self, b)
        }

        /// Calculates `self` + `rhs` + `carry` and returns a tuple containing
        /// the sum and the output carry.
        ///
        /// Performs "ternary addition" of two integer operands and a carry-in
        /// bit, and returns an output integer and a carry-out bit. This allows
        /// chaining together multiple additions to create a wider addition, and
        /// can be useful for bignum addition.
        ///
        #[doc = concat!("This can be thought of as a ", stringify!($BITS), "-bit \"full adder\", in the electronics sense.")]
        ///
        /// If the input carry is false, this method is equivalent to
        /// [`overflowing_add`](Self::overflowing_add), and the output carry is
        /// equal to the overflow flag. Note that although carry and overflow
        /// flags are similar for unsigned integers, they are different for
        /// signed integers.
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        ///
        #[doc = concat!("//    3  MAX    (a = 3 × 2^", stringify!($BITS), " + 2^", stringify!($BITS), " - 1)")]
        #[doc = concat!("// +  5    7    (b = 5 × 2^", stringify!($BITS), " + 7)")]
        /// // ---------
        #[doc = concat!("//    9    6    (sum = 9 × 2^", stringify!($BITS), " + 6)")]
        ///
        #[doc = concat!("let (a1, a0): (", stringify!($SelfT), ", ", stringify!($SelfT), ") = (3, ", stringify!($SelfT), "::MAX);")]
        #[doc = concat!("let (b1, b0): (", stringify!($SelfT), ", ", stringify!($SelfT), ") = (5, 7);")]
        /// let carry0 = false;
        ///
        /// let (sum0, carry1) = a0.carrying_add(b0, carry0);
        /// assert_eq!(carry1, true);
        /// let (sum1, carry2) = a1.carrying_add(b1, carry1);
        /// assert_eq!(carry2, false);
        ///
        /// assert_eq!((sum1, sum0), (9, 6));
        /// ```
        #[unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn carrying_add(self, rhs: Self, carry: bool) -> (Self, bool) {
            // note: longer-term this should be done via an intrinsic, but this has been shown
            //   to generate optimal code for now, and LLVM doesn't have an equivalent intrinsic
            let (a, b) = self.overflowing_add(rhs);
            let (c, d) = a.overflowing_add(carry as $SelfT);
            (c, b | d)
        }

        /// Calculates `self` + `rhs` with a signed `rhs`.
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
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".overflowing_add_signed(2), (3, false));")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".overflowing_add_signed(-2), (", stringify!($SelfT), "::MAX, true));")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).overflowing_add_signed(4), (1, true));")]
        /// ```
        #[stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[rustc_const_stable(feature = "mixed_integer_ops", since = "1.66.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn overflowing_add_signed(self, rhs: $SignedT) -> (Self, bool) {
            let (res, overflowed) = self.overflowing_add(rhs as Self);
            (res, overflowed ^ (rhs < 0))
        }

        /// Calculates `self` - `rhs`.
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
        #[doc = concat!("assert_eq!(5", stringify!($SelfT), ".overflowing_sub(2), (3, false));")]
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".overflowing_sub(1), (", stringify!($SelfT), "::MAX, true));")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
            let (a, b) = intrinsics::sub_with_overflow(self as $ActualT, rhs as $ActualT);
            (a as Self, b)
        }

        /// Calculates `self` &minus; `rhs` &minus; `borrow` and returns a tuple
        /// containing the difference and the output borrow.
        ///
        /// Performs "ternary subtraction" by subtracting both an integer
        /// operand and a borrow-in bit from `self`, and returns an output
        /// integer and a borrow-out bit. This allows chaining together multiple
        /// subtractions to create a wider subtraction, and can be useful for
        /// bignum subtraction.
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        ///
        #[doc = concat!("//    9    6    (a = 9 × 2^", stringify!($BITS), " + 6)")]
        #[doc = concat!("// -  5    7    (b = 5 × 2^", stringify!($BITS), " + 7)")]
        /// // ---------
        #[doc = concat!("//    3  MAX    (diff = 3 × 2^", stringify!($BITS), " + 2^", stringify!($BITS), " - 1)")]
        ///
        #[doc = concat!("let (a1, a0): (", stringify!($SelfT), ", ", stringify!($SelfT), ") = (9, 6);")]
        #[doc = concat!("let (b1, b0): (", stringify!($SelfT), ", ", stringify!($SelfT), ") = (5, 7);")]
        /// let borrow0 = false;
        ///
        /// let (diff0, borrow1) = a0.borrowing_sub(b0, borrow0);
        /// assert_eq!(borrow1, true);
        /// let (diff1, borrow2) = a1.borrowing_sub(b1, borrow1);
        /// assert_eq!(borrow2, false);
        ///
        #[doc = concat!("assert_eq!((diff1, diff0), (3, ", stringify!($SelfT), "::MAX));")]
        /// ```
        #[unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn borrowing_sub(self, rhs: Self, borrow: bool) -> (Self, bool) {
            // note: longer-term this should be done via an intrinsic, but this has been shown
            //   to generate optimal code for now, and LLVM doesn't have an equivalent intrinsic
            let (a, b) = self.overflowing_sub(rhs);
            let (c, d) = a.overflowing_sub(borrow as $SelfT);
            (c, b | d)
        }

        /// Calculates `self` - `rhs` with a signed `rhs`
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
        /// #![feature(mixed_integer_ops_unsigned_sub)]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".overflowing_sub_signed(2), (", stringify!($SelfT), "::MAX, true));")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".overflowing_sub_signed(-2), (3, false));")]
        #[doc = concat!("assert_eq!((", stringify!($SelfT), "::MAX - 2).overflowing_sub_signed(-4), (1, true));")]
        /// ```
        #[unstable(feature = "mixed_integer_ops_unsigned_sub", issue = "126043")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn overflowing_sub_signed(self, rhs: $SignedT) -> (Self, bool) {
            let (res, overflow) = self.overflowing_sub(rhs as Self);

            (res, overflow ^ (rhs < 0))
        }

        /// Computes the absolute difference between `self` and `other`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".abs_diff(80), 20", stringify!($SelfT), ");")]
        #[doc = concat!("assert_eq!(100", stringify!($SelfT), ".abs_diff(110), 10", stringify!($SelfT), ");")]
        /// ```
        #[stable(feature = "int_abs_diff", since = "1.60.0")]
        #[rustc_const_stable(feature = "int_abs_diff", since = "1.60.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn abs_diff(self, other: Self) -> Self {
            if mem::size_of::<Self>() == 1 {
                // Trick LLVM into generating the psadbw instruction when SSE2
                // is available and this function is autovectorized for u8's.
                (self as i32).wrapping_sub(other as i32).abs() as Self
            } else {
                if self < other {
                    other - self
                } else {
                    self - other
                }
            }
        }

        /// Calculates the multiplication of `self` and `rhs`.
        ///
        /// Returns a tuple of the multiplication along with a boolean
        /// indicating whether an arithmetic overflow would occur. If an
        /// overflow would have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// Please note that this example is shared between integer types.
        /// Which explains why `u32` is used here.
        ///
        /// ```
        /// assert_eq!(5u32.overflowing_mul(2), (10, false));
        /// assert_eq!(1_000_000_000u32.overflowing_mul(10), (1410065408, true));
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
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
        /// Which explains why `u32` is used here.
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// assert_eq!(5u32.widening_mul(2), (10, 0));
        /// assert_eq!(1_000_000_000u32.widening_mul(10), (1410065408, 2));
        /// ```
        #[unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[rustc_const_unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn widening_mul(self, rhs: Self) -> (Self, Self) {
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
        /// Which explains why `u32` is used here.
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// assert_eq!(5u32.carrying_mul(2, 0), (10, 0));
        /// assert_eq!(5u32.carrying_mul(2, 10), (20, 0));
        /// assert_eq!(1_000_000_000u32.carrying_mul(10, 0), (1410065408, 2));
        /// assert_eq!(1_000_000_000u32.carrying_mul(10, 10), (1410065418, 2));
        #[doc = concat!("assert_eq!(",
            stringify!($SelfT), "::MAX.carrying_mul(", stringify!($SelfT), "::MAX, ", stringify!($SelfT), "::MAX), ",
            "(0, ", stringify!($SelfT), "::MAX));"
        )]
        /// ```
        ///
        /// This is the core operation needed for scalar multiplication when
        /// implementing it for wider-than-native types.
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// fn scalar_mul_eq(little_endian_digits: &mut Vec<u16>, multiplicand: u16) {
        ///     let mut carry = 0;
        ///     for d in little_endian_digits.iter_mut() {
        ///         (*d, carry) = d.carrying_mul(multiplicand, carry);
        ///     }
        ///     if carry != 0 {
        ///         little_endian_digits.push(carry);
        ///     }
        /// }
        ///
        /// let mut v = vec![10, 20];
        /// scalar_mul_eq(&mut v, 3);
        /// assert_eq!(v, [30, 60]);
        ///
        /// assert_eq!(0x87654321_u64 * 0xFEED, 0x86D3D159E38D);
        /// let mut v = vec![0x4321, 0x8765];
        /// scalar_mul_eq(&mut v, 0xFEED);
        /// assert_eq!(v, [0xE38D, 0xD159, 0x86D3]);
        /// ```
        ///
        /// If `carry` is zero, this is similar to [`overflowing_mul`](Self::overflowing_mul),
        /// except that it gives the value of the overflow instead of just whether one happened:
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// let r = u8::carrying_mul(7, 13, 0);
        /// assert_eq!((r.0, r.1 != 0), u8::overflowing_mul(7, 13));
        /// let r = u8::carrying_mul(13, 42, 0);
        /// assert_eq!((r.0, r.1 != 0), u8::overflowing_mul(13, 42));
        /// ```
        ///
        /// The value of the first field in the returned tuple matches what you'd get
        /// by combining the [`wrapping_mul`](Self::wrapping_mul) and
        /// [`wrapping_add`](Self::wrapping_add) methods:
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// assert_eq!(
        ///     789_u16.carrying_mul(456, 123).0,
        ///     789_u16.wrapping_mul(456).wrapping_add(123),
        /// );
        /// ```
        #[unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[rustc_const_unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn carrying_mul(self, rhs: Self, carry: Self) -> (Self, Self) {
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
        /// Which explains why `u32` is used here.
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// assert_eq!(5u32.carrying_mul_add(2, 0, 0), (10, 0));
        /// assert_eq!(5u32.carrying_mul_add(2, 10, 10), (30, 0));
        /// assert_eq!(1_000_000_000u32.carrying_mul_add(10, 0, 0), (1410065408, 2));
        /// assert_eq!(1_000_000_000u32.carrying_mul_add(10, 10, 10), (1410065428, 2));
        #[doc = concat!("assert_eq!(",
            stringify!($SelfT), "::MAX.carrying_mul_add(", stringify!($SelfT), "::MAX, ", stringify!($SelfT), "::MAX, ", stringify!($SelfT), "::MAX), ",
            "(", stringify!($SelfT), "::MAX, ", stringify!($SelfT), "::MAX));"
        )]
        /// ```
        #[unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[rustc_const_unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn carrying_mul_add(self, rhs: Self, carry: Self, add: Self) -> (Self, Self) {
            intrinsics::carrying_mul_add(self, rhs, carry, add)
        }

        /// Calculates the divisor when `self` is divided by `rhs`.
        ///
        /// Returns a tuple of the divisor along with a boolean indicating
        /// whether an arithmetic overflow would occur. Note that for unsigned
        /// integers overflow never occurs, so the second value is always
        /// `false`.
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
        /// ```
        #[inline(always)]
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_overflowing_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[track_caller]
        pub const fn overflowing_div(self, rhs: Self) -> (Self, bool) {
            (self / rhs, false)
        }

        /// Calculates the quotient of Euclidean division `self.div_euclid(rhs)`.
        ///
        /// Returns a tuple of the divisor along with a boolean indicating
        /// whether an arithmetic overflow would occur. Note that for unsigned
        /// integers overflow never occurs, so the second value is always
        /// `false`.
        /// Since, for the positive integers, all common
        /// definitions of division are equal, this
        /// is exactly equal to `self.overflowing_div(rhs)`.
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
        /// ```
        #[inline(always)]
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[track_caller]
        pub const fn overflowing_div_euclid(self, rhs: Self) -> (Self, bool) {
            (self / rhs, false)
        }

        /// Calculates the remainder when `self` is divided by `rhs`.
        ///
        /// Returns a tuple of the remainder after dividing along with a boolean
        /// indicating whether an arithmetic overflow would occur. Note that for
        /// unsigned integers overflow never occurs, so the second value is
        /// always `false`.
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
        /// ```
        #[inline(always)]
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_overflowing_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[track_caller]
        pub const fn overflowing_rem(self, rhs: Self) -> (Self, bool) {
            (self % rhs, false)
        }

        /// Calculates the remainder `self.rem_euclid(rhs)` as if by Euclidean division.
        ///
        /// Returns a tuple of the modulo after dividing along with a boolean
        /// indicating whether an arithmetic overflow would occur. Note that for
        /// unsigned integers overflow never occurs, so the second value is
        /// always `false`.
        /// Since, for the positive integers, all common
        /// definitions of division are equal, this operation
        /// is exactly equal to `self.overflowing_rem(rhs)`.
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
        /// ```
        #[inline(always)]
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[track_caller]
        pub const fn overflowing_rem_euclid(self, rhs: Self) -> (Self, bool) {
            (self % rhs, false)
        }

        /// Negates self in an overflowing fashion.
        ///
        /// Returns `!self + 1` using wrapping operations to return the value
        /// that represents the negation of this unsigned value. Note that for
        /// positive unsigned values overflow always occurs, but negating 0 does
        /// not overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".overflowing_neg(), (0, false));")]
        #[doc = concat!("assert_eq!(2", stringify!($SelfT), ".overflowing_neg(), (-2i32 as ", stringify!($SelfT), ", true));")]
        /// ```
        #[inline(always)]
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        pub const fn overflowing_neg(self) -> (Self, bool) {
            ((!self).wrapping_add(1), self != 0)
        }

        /// Shifts self left by `rhs` bits.
        ///
        /// Returns a tuple of the shifted version of self along with a boolean
        /// indicating whether the shift value was larger than or equal to the
        /// number of bits. If the shift value is too large, then value is
        /// masked (N-1) where N is the number of bits, and this value is then
        /// used to perform the shift.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0x1", stringify!($SelfT), ".overflowing_shl(4), (0x10, false));")]
        #[doc = concat!("assert_eq!(0x1", stringify!($SelfT), ".overflowing_shl(132), (0x10, true));")]
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".overflowing_shl(", stringify!($BITS_MINUS_ONE), "), (0, false));")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn overflowing_shl(self, rhs: u32) -> (Self, bool) {
            (self.wrapping_shl(rhs), rhs >= Self::BITS)
        }

        /// Shifts self right by `rhs` bits.
        ///
        /// Returns a tuple of the shifted version of self along with a boolean
        /// indicating whether the shift value was larger than or equal to the
        /// number of bits. If the shift value is too large, then value is
        /// masked (N-1) where N is the number of bits, and this value is then
        /// used to perform the shift.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".overflowing_shr(4), (0x1, false));")]
        #[doc = concat!("assert_eq!(0x10", stringify!($SelfT), ".overflowing_shr(132), (0x1, true));")]
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        pub const fn overflowing_shr(self, rhs: u32) -> (Self, bool) {
            (self.wrapping_shr(rhs), rhs >= Self::BITS)
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
        #[doc = concat!("assert_eq!(3", stringify!($SelfT), ".overflowing_pow(5), (243, false));")]
        /// assert_eq!(3u8.overflowing_pow(6), (217, true));
        /// ```
        #[stable(feature = "no_panic_pow", since = "1.34.0")]
        #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn overflowing_pow(self, mut exp: u32) -> (Self, bool) {
            if exp == 0{
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
        #[doc = concat!("assert_eq!(2", stringify!($SelfT), ".pow(5), 32);")]
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
        pub const fn isqrt(self) -> Self {
            let result = crate::num::int_sqrt::$ActualT(self as $ActualT) as $SelfT;

            // Inform the optimizer what the range of outputs is. If testing
            // `core` crashes with no panic message and a `num::int_sqrt::u*`
            // test failed, it's because your edits caused these assertions or
            // the assertions in `fn isqrt` of `nonzero.rs` to become false.
            //
            // SAFETY: Integer square root is a monotonically nondecreasing
            // function, which means that increasing the input will never
            // cause the output to decrease. Thus, since the input for unsigned
            // integers is bounded by `[0, <$ActualT>::MAX]`, sqrt(n) will be
            // bounded by `[sqrt(0), sqrt(<$ActualT>::MAX)]`.
            unsafe {
                const MAX_RESULT: $SelfT = crate::num::int_sqrt::$ActualT(<$ActualT>::MAX) as $SelfT;
                crate::hint::assert_unchecked(result <= MAX_RESULT);
            }

            result
        }

        /// Performs Euclidean division.
        ///
        /// Since, for the positive integers, all common
        /// definitions of division are equal, this
        /// is exactly equal to `self / rhs`.
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
        #[doc = concat!("assert_eq!(7", stringify!($SelfT), ".div_euclid(4), 1); // or any other integer type")]
        /// ```
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[track_caller]
        pub const fn div_euclid(self, rhs: Self) -> Self {
            self / rhs
        }


        /// Calculates the least remainder of `self (mod rhs)`.
        ///
        /// Since, for the positive integers, all common
        /// definitions of division are equal, this
        /// is exactly equal to `self % rhs`.
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
        #[doc = concat!("assert_eq!(7", stringify!($SelfT), ".rem_euclid(4), 3); // or any other integer type")]
        /// ```
        #[doc(alias = "modulo", alias = "mod")]
        #[stable(feature = "euclidean_division", since = "1.38.0")]
        #[rustc_const_stable(feature = "const_euclidean_int_methods", since = "1.52.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[track_caller]
        pub const fn rem_euclid(self, rhs: Self) -> Self {
            self % rhs
        }

        /// Calculates the quotient of `self` and `rhs`, rounding the result towards negative infinity.
        ///
        /// This is the same as performing `self / rhs` for all unsigned integers.
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
        /// #![feature(int_roundings)]
        #[doc = concat!("assert_eq!(7_", stringify!($SelfT), ".div_floor(4), 1);")]
        /// ```
        #[unstable(feature = "int_roundings", issue = "88581")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline(always)]
        #[track_caller]
        pub const fn div_floor(self, rhs: Self) -> Self {
            self / rhs
        }

        /// Calculates the quotient of `self` and `rhs`, rounding the result towards positive infinity.
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
        #[doc = concat!("assert_eq!(7_", stringify!($SelfT), ".div_ceil(4), 2);")]
        /// ```
        #[stable(feature = "int_roundings1", since = "1.73.0")]
        #[rustc_const_stable(feature = "int_roundings1", since = "1.73.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[track_caller]
        pub const fn div_ceil(self, rhs: Self) -> Self {
            let d = self / rhs;
            let r = self % rhs;
            if r > 0 {
                d + 1
            } else {
                d
            }
        }

        /// Calculates the smallest value greater than or equal to `self` that
        /// is a multiple of `rhs`.
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
        #[doc = concat!("assert_eq!(16_", stringify!($SelfT), ".next_multiple_of(8), 16);")]
        #[doc = concat!("assert_eq!(23_", stringify!($SelfT), ".next_multiple_of(8), 24);")]
        /// ```
        #[stable(feature = "int_roundings1", since = "1.73.0")]
        #[rustc_const_stable(feature = "int_roundings1", since = "1.73.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[rustc_inherit_overflow_checks]
        pub const fn next_multiple_of(self, rhs: Self) -> Self {
            match self % rhs {
                0 => self,
                r => self + (rhs - r)
            }
        }

        /// Calculates the smallest value greater than or equal to `self` that
        /// is a multiple of `rhs`. Returns `None` if `rhs` is zero or the
        /// operation would result in overflow.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(16_", stringify!($SelfT), ".checked_next_multiple_of(8), Some(16));")]
        #[doc = concat!("assert_eq!(23_", stringify!($SelfT), ".checked_next_multiple_of(8), Some(24));")]
        #[doc = concat!("assert_eq!(1_", stringify!($SelfT), ".checked_next_multiple_of(0), None);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.checked_next_multiple_of(2), None);")]
        /// ```
        #[stable(feature = "int_roundings1", since = "1.73.0")]
        #[rustc_const_stable(feature = "int_roundings1", since = "1.73.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn checked_next_multiple_of(self, rhs: Self) -> Option<Self> {
            match try_opt!(self.checked_rem(rhs)) {
                0 => Some(self),
                // rhs - r cannot overflow because r is smaller than rhs
                r => self.checked_add(rhs - r)
            }
        }

        /// Returns `true` if `self` is an integer multiple of `rhs`, and false otherwise.
        ///
        /// This function is equivalent to `self % rhs == 0`, except that it will not panic
        /// for `rhs == 0`. Instead, `0.is_multiple_of(0) == true`, and for any non-zero `n`,
        /// `n.is_multiple_of(0) == false`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(unsigned_is_multiple_of)]
        #[doc = concat!("assert!(6_", stringify!($SelfT), ".is_multiple_of(2));")]
        #[doc = concat!("assert!(!5_", stringify!($SelfT), ".is_multiple_of(2));")]
        ///
        #[doc = concat!("assert!(0_", stringify!($SelfT), ".is_multiple_of(0));")]
        #[doc = concat!("assert!(!6_", stringify!($SelfT), ".is_multiple_of(0));")]
        /// ```
        #[unstable(feature = "unsigned_is_multiple_of", issue = "128101")]
        #[must_use]
        #[inline]
        #[rustc_inherit_overflow_checks]
        pub const fn is_multiple_of(self, rhs: Self) -> bool {
            match rhs {
                0 => self == 0,
                _ => self % rhs == 0,
            }
        }

        /// Returns `true` if and only if `self == 2^k` for some `k`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert!(16", stringify!($SelfT), ".is_power_of_two());")]
        #[doc = concat!("assert!(!10", stringify!($SelfT), ".is_power_of_two());")]
        /// ```
        #[must_use]
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_is_power_of_two", since = "1.32.0")]
        #[inline(always)]
        pub const fn is_power_of_two(self) -> bool {
            self.count_ones() == 1
        }

        // Returns one less than next power of two.
        // (For 8u8 next power of two is 8u8 and for 6u8 it is 8u8)
        //
        // 8u8.one_less_than_next_power_of_two() == 7
        // 6u8.one_less_than_next_power_of_two() == 7
        //
        // This method cannot overflow, as in the `next_power_of_two`
        // overflow cases it instead ends up returning the maximum value
        // of the type, and can return 0 for 0.
        #[inline]
        const fn one_less_than_next_power_of_two(self) -> Self {
            if self <= 1 { return 0; }

            let p = self - 1;
            // SAFETY: Because `p > 0`, it cannot consist entirely of leading zeros.
            // That means the shift is always in-bounds, and some processors
            // (such as intel pre-haswell) have more efficient ctlz
            // intrinsics when the argument is non-zero.
            let z = unsafe { intrinsics::ctlz_nonzero(p) };
            <$SelfT>::MAX >> z
        }

        /// Returns the smallest power of two greater than or equal to `self`.
        ///
        /// When return value overflows (i.e., `self > (1 << (N-1))` for type
        /// `uN`), it panics in debug mode and the return value is wrapped to 0 in
        /// release mode (the only situation in which this method can return 0).
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(2", stringify!($SelfT), ".next_power_of_two(), 2);")]
        #[doc = concat!("assert_eq!(3", stringify!($SelfT), ".next_power_of_two(), 4);")]
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".next_power_of_two(), 1);")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        #[rustc_inherit_overflow_checks]
        pub const fn next_power_of_two(self) -> Self {
            self.one_less_than_next_power_of_two() + 1
        }

        /// Returns the smallest power of two greater than or equal to `self`. If
        /// the next power of two is greater than the type's maximum value,
        /// `None` is returned, otherwise the power of two is wrapped in `Some`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("assert_eq!(2", stringify!($SelfT), ".checked_next_power_of_two(), Some(2));")]
        #[doc = concat!("assert_eq!(3", stringify!($SelfT), ".checked_next_power_of_two(), Some(4));")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.checked_next_power_of_two(), None);")]
        /// ```
        #[inline]
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        pub const fn checked_next_power_of_two(self) -> Option<Self> {
            self.one_less_than_next_power_of_two().checked_add(1)
        }

        /// Returns the smallest power of two greater than or equal to `n`. If
        /// the next power of two is greater than the type's maximum value,
        /// the return value is wrapped to `0`.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        /// #![feature(wrapping_next_power_of_two)]
        ///
        #[doc = concat!("assert_eq!(2", stringify!($SelfT), ".wrapping_next_power_of_two(), 2);")]
        #[doc = concat!("assert_eq!(3", stringify!($SelfT), ".wrapping_next_power_of_two(), 4);")]
        #[doc = concat!("assert_eq!(", stringify!($SelfT), "::MAX.wrapping_next_power_of_two(), 0);")]
        /// ```
        #[inline]
        #[unstable(feature = "wrapping_next_power_of_two", issue = "32463",
                   reason = "needs decision on wrapping behavior")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        pub const fn wrapping_next_power_of_two(self) -> Self {
            self.one_less_than_next_power_of_two().wrapping_add(1)
        }

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
        pub const fn to_be_bytes(self) -> [u8; mem::size_of::<Self>()] {
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
        pub const fn to_le_bytes(self) -> [u8; mem::size_of::<Self>()] {
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
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        // SAFETY: const sound because integers are plain old datatypes so we can always
        // transmute them to arrays of bytes
        #[inline]
        pub const fn to_ne_bytes(self) -> [u8; mem::size_of::<Self>()] {
            // SAFETY: integers are plain old datatypes so we can always transmute them to
            // arrays of bytes
            unsafe { mem::transmute(self) }
        }

        /// Creates a native endian integer value from its representation
        /// as a byte array in big endian.
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
        #[doc = concat!("    let (int_bytes, rest) = input.split_at(std::mem::size_of::<", stringify!($SelfT), ">());")]
        ///     *input = rest;
        #[doc = concat!("    ", stringify!($SelfT), "::from_be_bytes(int_bytes.try_into().unwrap())")]
        /// }
        /// ```
        #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
        #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
        #[must_use]
        #[inline]
        pub const fn from_be_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
            Self::from_be(Self::from_ne_bytes(bytes))
        }

        /// Creates a native endian integer value from its representation
        /// as a byte array in little endian.
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
        #[doc = concat!("    let (int_bytes, rest) = input.split_at(std::mem::size_of::<", stringify!($SelfT), ">());")]
        ///     *input = rest;
        #[doc = concat!("    ", stringify!($SelfT), "::from_le_bytes(int_bytes.try_into().unwrap())")]
        /// }
        /// ```
        #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
        #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
        #[must_use]
        #[inline]
        pub const fn from_le_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
            Self::from_le(Self::from_ne_bytes(bytes))
        }

        /// Creates a native endian integer value from its memory representation
        /// as a byte array in native endianness.
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
        #[doc = concat!("    ", $be_bytes, "")]
        /// } else {
        #[doc = concat!("    ", $le_bytes, "")]
        /// });
        #[doc = concat!("assert_eq!(value, ", $swap_op, ");")]
        /// ```
        ///
        /// When starting from a slice rather than an array, fallible conversion APIs can be used:
        ///
        /// ```
        #[doc = concat!("fn read_ne_", stringify!($SelfT), "(input: &mut &[u8]) -> ", stringify!($SelfT), " {")]
        #[doc = concat!("    let (int_bytes, rest) = input.split_at(std::mem::size_of::<", stringify!($SelfT), ">());")]
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
        pub const fn from_ne_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
            // SAFETY: integers are plain old datatypes so we can always transmute to them
            unsafe { mem::transmute(bytes) }
        }

        /// New code should prefer to use
        #[doc = concat!("[`", stringify!($SelfT), "::MIN", "`] instead.")]
        ///
        /// Returns the smallest value that can be represented by this integer type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_promotable]
        #[inline(always)]
        #[rustc_const_stable(feature = "const_max_value", since = "1.32.0")]
        #[deprecated(since = "TBD", note = "replaced by the `MIN` associated constant on this type")]
        #[rustc_diagnostic_item = concat!(stringify!($SelfT), "_legacy_fn_min_value")]
        pub const fn min_value() -> Self { Self::MIN }

        /// New code should prefer to use
        #[doc = concat!("[`", stringify!($SelfT), "::MAX", "`] instead.")]
        ///
        /// Returns the largest value that can be represented by this integer type.
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_promotable]
        #[inline(always)]
        #[rustc_const_stable(feature = "const_max_value", since = "1.32.0")]
        #[deprecated(since = "TBD", note = "replaced by the `MAX` associated constant on this type")]
        #[rustc_diagnostic_item = concat!(stringify!($SelfT), "_legacy_fn_max_value")]
        pub const fn max_value() -> Self { Self::MAX }
    }
}
